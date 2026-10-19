#!/usr/bin/env python3
"""Ollama API proxy: translates llama.cpp responses to Ollama format.

Sits between clients and llama-server:
  Client -> :11434 (this proxy) -> :8081 (llama-server)

Translated:   /api/chat, /api/generate, /api/tags, /api/show, /api/version, /api/ps
Passthrough:  everything else (/v1/*, /health, /completion, ...). SSE streams are
              forwarded as they arrive instead of being buffered.

Environment:
  OLLAMA_PROXY_HOST      bind address (default 127.0.0.1; use 0.0.0.0 to expose on the LAN)
  OLLAMA_PROXY_TIMEOUT   backend timeout in seconds (default 300)
  OLLAMA_PROXY_MAX_BODY  max request body in bytes (default 33554432)

Not supported: per-request num_ctx (llama-server fixes the context at startup),
keep_alive, raw/template/suffix.
"""

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKEND_PORT = 8081
LISTEN_PORT = 11434
LISTEN_HOST = os.environ.get("OLLAMA_PROXY_HOST", "127.0.0.1")
BACKEND_TIMEOUT = int(os.environ.get("OLLAMA_PROXY_TIMEOUT", "300"))
MAX_BODY = int(os.environ.get("OLLAMA_PROXY_MAX_BODY", str(32 * 1024 * 1024)))

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "seed", "stop",
                 "repeat_penalty", "presence_penalty", "frequency_penalty")


class BackendError(Exception):
    def __init__(self, code, body):
        super().__init__(f"backend error {code}")
        self.code = code
        self.body = body


# --- pure helpers (no I/O) ---

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def collect_options(req):
    """Sampling options from the top level and from req['options'] (options win)."""
    keys = SAMPLING_KEYS + ("num_predict",)
    opts = {k: req[k] for k in keys if k in req}
    options = req.get("options")
    if isinstance(options, dict):
        opts.update({k: options[k] for k in keys if k in options})
    return opts


def map_finish(reason):
    if reason in (None, "", "stop", "tool_calls", "function_call"):
        return "stop"
    return reason


def to_openai_messages(messages):
    """Ollama chat messages -> OpenAI chat messages."""
    out = []
    for msg in messages or []:
        m = dict(msg)
        m.pop("thinking", None)
        images = m.pop("images", None)
        if images:
            content = [{"type": "text", "text": m.get("content") or ""}]
            for img in images:
                content.append({"type": "image_url",
                                "image_url": {"url": "data:image/jpeg;base64," + img}})
            m["content"] = content
        calls = m.get("tool_calls")
        if calls:
            converted = []
            for i, tc in enumerate(calls):
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args)
                converted.append({"id": tc.get("id", f"call_{i}"), "type": "function",
                                  "function": {"name": fn.get("name", ""), "arguments": args}})
            m["tool_calls"] = converted
        out.append(m)
    return out


def convert_tool_calls(calls):
    """OpenAI tool_calls (arguments = JSON string) -> Ollama (arguments = object)."""
    out = []
    for tc in calls or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return out


def chat_message_from_openai(message):
    out = {"role": message.get("role", "assistant"), "content": message.get("content") or ""}
    if message.get("reasoning_content"):
        out["thinking"] = message["reasoning_content"]
    calls = convert_tool_calls(message.get("tool_calls"))
    if calls:
        out["tool_calls"] = calls
    return out


def response_format(fmt):
    if fmt == "json":
        return {"type": "json_object"}
    if isinstance(fmt, dict):
        return {"type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": fmt}}
    return None


def error_message(err):
    """llama-server errors are {"error": {"message": ...}}; Ollama uses {"error": "..."}."""
    if isinstance(err, dict):
        return str(err.get("message") or json.dumps(err))
    return str(err)


def normalize_error(payload, code):
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": (payload or "").strip() or f"backend error {code}"})
    if isinstance(data, dict) and data.get("error"):
        return json.dumps({"error": error_message(data["error"])})
    return payload


def sse_events(resp):
    """Yield the payload of each 'data:' line of an SSE response."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            yield line[5:].strip()


class OllamaProxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ollama-rdna1-proxy"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.command} {self.path}\n")

    # --- GET ---

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/":
            self._respond(200, "application/json",
                          json.dumps({"status": "Ollama is running (ollama-rdna1)"}))
            return

        if path == "/api/tags":
            resp = self._forward("GET", "/v1/models")
            if resp is None:
                return
            try:
                data = json.loads(resp)
                models = data.get("models", data.get("data", []))
                ollama_models = [{
                    "name": m.get("name", m.get("id", "unknown")),
                    "model": m.get("model", m.get("id", "unknown")),
                    "modified_at": m.get("modified_at", ""),
                    "size": m.get("size", 0),
                    "digest": m.get("digest", ""),
                    "details": m.get("details", {}),
                } for m in models]
                self._respond(200, "application/json", json.dumps({"models": ollama_models}))
            except (json.JSONDecodeError, AttributeError):
                self._respond(200, "application/json", resp)
            return

        if path == "/api/version":
            self._respond(200, "application/json", json.dumps({"version": "ollama-rdna1-0.1.0"}))
            return

        if path == "/api/ps":
            self._respond(200, "application/json", json.dumps({"models": []}))
            return

        self._passthrough("GET", self.path)

    # --- POST ---

    def do_POST(self):
        body = self._read_body()
        if body is None:
            return

        path = self.path.split("?", 1)[0]
        if path == "/api/chat":
            self._handle_chat(body)
        elif path == "/api/generate":
            self._handle_generate(body)
        elif path == "/api/show":
            self._handle_show(body)
        else:
            self._passthrough("POST", self.path, body)

    # --- /api/chat ---

    def _handle_chat(self, body):
        req = self._parse_json(body)
        if req is None:
            return

        stream = req.get("stream", True)
        model_name = req.get("model", "")

        llama_req = {
            "messages": to_openai_messages(req.get("messages", [])),
            "model": model_name,
            "stream": bool(stream),
        }
        if stream:
            llama_req["stream_options"] = {"include_usage": True}

        opts = collect_options(req)
        num_predict = opts.pop("num_predict", None)
        llama_req.update(opts)
        if isinstance(num_predict, int) and num_predict > 0:
            llama_req["max_tokens"] = num_predict
        if req.get("tools"):
            llama_req["tools"] = req["tools"]
        fmt = response_format(req.get("format"))
        if fmt:
            llama_req["response_format"] = fmt

        t0 = time.time_ns()
        if stream:
            self._stream_chat(llama_req, model_name, t0)
            return

        resp = self._forward("POST", "/v1/chat/completions", json.dumps(llama_req))
        if resp is None:
            return
        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            self._error(502, "bad backend response")
            return

        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        self._respond(200, "application/json", json.dumps({
            "model": data.get("model", model_name),
            "created_at": now_iso(),
            "message": chat_message_from_openai(choice.get("message") or {}),
            "done_reason": map_finish(choice.get("finish_reason")),
            "done": True,
            "total_duration": time.time_ns() - t0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }))

    def _stream_chat(self, llama_req, model_name, t0):
        try:
            up = self._backend_open("POST", "/v1/chat/completions", json.dumps(llama_req))
        except BackendError as e:
            self._respond_backend_error(e)
            return

        created = now_iso()
        resp_model = model_name
        tool_acc = {}
        finish = None
        usage = {}
        started = False
        try:
            with up:
                for event in sse_events(up):
                    if event == "[DONE]":
                        break
                    try:
                        data = json.loads(event)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        # llama-server reports failures as an SSE event inside a 200 response
                        message = error_message(data["error"])
                        if started:
                            self._write_stream_error(message)
                        else:
                            self._error(500, message)
                        return
                    resp_model = data.get("model") or resp_model
                    if data.get("usage"):
                        usage = data["usage"]
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    for tc in delta.get("tool_calls") or []:
                        acc = tool_acc.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        if fn.get("name") and not acc["name"]:
                            acc["name"] = fn["name"]
                        if fn.get("arguments"):
                            acc["arguments"] += fn["arguments"]
                    message = {"role": "assistant", "content": delta.get("content") or ""}
                    if delta.get("reasoning_content"):
                        message["thinking"] = delta["reasoning_content"]
                    if message["content"] or message.get("thinking"):
                        if not started:
                            self._begin_ndjson()
                            started = True
                        self._write_json_line({"model": resp_model, "created_at": created,
                                               "message": message, "done": False})
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]

            if finish is None:
                message = "backend closed the stream without finishing the response"
                if started:
                    self._write_stream_error(message)
                else:
                    self._error(502, message)
                return

            if not started:
                self._begin_ndjson()
            final_msg = {"role": "assistant", "content": ""}
            if tool_acc:
                final_msg["tool_calls"] = convert_tool_calls(
                    [{"function": tool_acc[i]} for i in sorted(tool_acc)])
            self._write_json_line({
                "model": resp_model, "created_at": created, "message": final_msg,
                "done_reason": map_finish(finish), "done": True,
                "total_duration": time.time_ns() - t0,
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
            })
            self._end_chunks()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (OSError, ValueError) as e:
            if started:
                self._write_stream_error(f"backend stream failed: {e}")
            else:
                self._error(502, f"backend stream failed: {e}")

    # --- /api/generate ---

    def _handle_generate(self, body):
        req = self._parse_json(body)
        if req is None:
            return

        stream = req.get("stream", True)
        model_name = req.get("model", "")

        prompt = req.get("prompt", "")
        if req.get("system"):
            prompt = req["system"] + "\n" + prompt
        llama_req = {"prompt": prompt, "stream": bool(stream)}

        opts = collect_options(req)
        num_predict = opts.pop("num_predict", None)
        llama_req.update(opts)
        # Ollama's default is "until done"; llama.cpp spells that -1.
        llama_req["n_predict"] = num_predict if isinstance(num_predict, int) and num_predict > 0 else -1
        fmt = req.get("format")
        if fmt == "json":
            llama_req["json_schema"] = {"type": "object"}
        elif isinstance(fmt, dict):
            llama_req["json_schema"] = fmt

        t0 = time.time_ns()
        if stream:
            self._stream_generate(llama_req, model_name, t0)
            return

        resp = self._forward("POST", "/completion", json.dumps(llama_req))
        if resp is None:
            return
        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            self._error(502, "bad backend response")
            return

        self._respond(200, "application/json", json.dumps({
            "model": model_name,
            "created_at": now_iso(),
            "response": data.get("content", ""),
            "done": True,
            "done_reason": "length" if data.get("stop_type") == "limit" else "stop",
            "total_duration": time.time_ns() - t0,
            "prompt_eval_count": data.get("tokens_evaluated", 0),
            "eval_count": data.get("tokens_predicted", 0),
        }))

    def _stream_generate(self, llama_req, model_name, t0):
        try:
            up = self._backend_open("POST", "/completion", json.dumps(llama_req))
        except BackendError as e:
            self._respond_backend_error(e)
            return

        created = now_iso()
        last = {}
        started = False
        try:
            with up:
                for event in sse_events(up):
                    try:
                        data = json.loads(event)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        message = error_message(data["error"])
                        if started:
                            self._write_stream_error(message)
                        else:
                            self._error(500, message)
                        return
                    last = data
                    if data.get("content"):
                        if not started:
                            self._begin_ndjson()
                            started = True
                        self._write_json_line({"model": model_name, "created_at": created,
                                               "response": data["content"], "done": False})
                    if data.get("stop"):
                        break

            if not last.get("stop"):
                message = "backend closed the stream without finishing the response"
                if started:
                    self._write_stream_error(message)
                else:
                    self._error(502, message)
                return

            if not started:
                self._begin_ndjson()
            self._write_json_line({
                "model": model_name, "created_at": created, "response": "",
                "done": True,
                "done_reason": "length" if last.get("stop_type") == "limit" else "stop",
                "total_duration": time.time_ns() - t0,
                "prompt_eval_count": last.get("tokens_evaluated", 0),
                "eval_count": last.get("tokens_predicted", 0),
            })
            self._end_chunks()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (OSError, ValueError) as e:
            if started:
                self._write_stream_error(f"backend stream failed: {e}")
            else:
                self._error(502, f"backend stream failed: {e}")

    # --- /api/show ---

    def _handle_show(self, body):
        """llama-server has no /api/show; build a minimal answer from /props."""
        props = self._backend_json("GET", "/props") or {}
        settings = props.get("default_generation_settings") or {}
        n_ctx = settings.get("n_ctx") or props.get("n_ctx") or 0
        modalities = props.get("modalities") or {}
        capabilities = ["completion"]
        if modalities.get("vision"):
            capabilities.append("vision")
        self._respond(200, "application/json", json.dumps({
            "license": "", "modelfile": "", "parameters": "", "template": "",
            "details": {"parent_model": "", "format": "gguf", "family": "llama",
                        "families": ["llama"], "parameter_size": "", "quantization_level": ""},
            "model_info": {"general.architecture": "llama", "llama.context_length": n_ctx},
            "capabilities": capabilities,
        }))

    # --- passthrough (keeps SSE streaming) ---

    def _passthrough(self, method, path, body=None):
        try:
            up = self._backend_open(method, path, body)
        except BackendError as e:
            self._respond(e.code, "application/json", e.body)
            return

        with up:
            ctype = up.headers.get("Content-Type", "application/json")
            if "text/event-stream" not in ctype:
                try:
                    data = up.read()
                except socket.timeout:
                    self._error(504, "backend timeout")
                    return
                except OSError:
                    self._error(502, "backend read failed")
                    return
                self._respond(up.status, ctype, data)
                return

            self.send_response(up.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    chunk = up.read1(8192)
                    if not chunk:
                        break
                    self._write_chunk(chunk)
                self._end_chunks()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close_connection = True

    # --- chunked NDJSON helpers ---

    def _begin_ndjson(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _write_json_line(self, obj):
        self._write_chunk(json.dumps(obj) + "\n")

    def _write_chunk(self, data):
        encoded = data.encode() if isinstance(data, str) else data
        if not encoded:
            return
        self.wfile.write(f"{len(encoded):x}\r\n".encode())
        self.wfile.write(encoded)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _end_chunks(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _write_stream_error(self, message):
        """Headers are already sent; report the failure as Ollama does and end the stream."""
        try:
            self._write_json_line({"error": message})
            self._end_chunks()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    # --- request helpers ---

    def _read_body(self):
        """Return the request body as str, or None after having sent an error."""
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            return self._read_chunked_body()

        raw = (self.headers.get("Content-Length") or "0").strip()
        try:
            length = int(raw)
            if length < 0:
                raise ValueError
        except ValueError:
            self.close_connection = True
            self._error(400, "invalid Content-Length")
            return None
        if length > MAX_BODY:
            self.close_connection = True
            self._error(413, "request body too large")
            return None
        return self.rfile.read(length).decode("utf-8", "replace") if length else ""

    def _read_chunked_body(self):
        chunks = []
        total = 0
        while True:
            line = self.rfile.readline(64).strip()
            try:
                size = int(line.split(b";")[0], 16)
            except ValueError:
                self.close_connection = True
                self._error(400, "invalid chunked body")
                return None
            if size == 0:
                self.rfile.readline(64)  # final CRLF (trailers are ignored)
                break
            total += size
            if total > MAX_BODY:
                self.close_connection = True
                self._error(413, "request body too large")
                return None
            chunks.append(self.rfile.read(size))
            self.rfile.readline(64)  # CRLF after the chunk
        return b"".join(chunks).decode("utf-8", "replace")

    def _parse_json(self, body):
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._error(400, "invalid JSON")
            return None
        if not isinstance(req, dict):
            self._error(400, "JSON object expected")
            return None
        return req

    # --- backend helpers ---

    def _backend_open(self, method, path, body=None):
        url = f"http://127.0.0.1:{BACKEND_PORT}{path}"
        data = body.encode("utf-8") if isinstance(body, str) and body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            return urllib.request.urlopen(req, timeout=BACKEND_TIMEOUT)
        except urllib.error.HTTPError as e:
            try:
                payload = e.read().decode("utf-8", "replace")
            except OSError:
                payload = ""
            raise BackendError(e.code, payload or json.dumps({"error": str(e)}))
        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout):
                raise BackendError(504, json.dumps({"error": "backend timeout"}))
            raise BackendError(502, json.dumps({"error": "backend not reachable"}))
        except socket.timeout:
            raise BackendError(504, json.dumps({"error": "backend timeout"}))
        except OSError:
            raise BackendError(502, json.dumps({"error": "backend not reachable"}))

    def _forward(self, method, path, body=None):
        """Buffered request. On failure the error is sent to the client and None is returned."""
        try:
            with self._backend_open(method, path, body) as up:
                return up.read().decode("utf-8", "replace")
        except BackendError as e:
            self._respond_backend_error(e)
        except socket.timeout:
            self._error(504, "backend timeout")
        except OSError:
            self._error(502, "backend read failed")
        return None

    def _backend_json(self, method, path):
        """Quiet variant: parsed JSON or None, nothing is sent to the client."""
        try:
            with self._backend_open(method, path) as up:
                return json.loads(up.read().decode("utf-8", "replace"))
        except (BackendError, OSError, ValueError):
            return None

    # --- responses ---

    def _respond_backend_error(self, e):
        self._respond(e.code, "application/json", normalize_error(e.body, e.code))

    def _error(self, code, message):
        self._respond(code, "application/json", json.dumps({"error": message}))

    def _respond(self, code, content_type, body):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)


def main():
    global BACKEND_PORT
    port = int(sys.argv[1]) if len(sys.argv) > 1 else LISTEN_PORT
    BACKEND_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else BACKEND_PORT

    server = ThreadingHTTPServer((LISTEN_HOST, port), OllamaProxy)
    server.daemon_threads = True
    print(f"ollama-proxy listening on {LISTEN_HOST}:{port} -> backend :{BACKEND_PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
