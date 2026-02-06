# Gunicorn configuration for Svend

import multiprocessing

# Bind to localhost (Cloudflare Tunnel will handle external)
bind = "127.0.0.1:8000"

# Workers: 2-4 x CPU cores for CPU-bound, more for I/O-bound
workers = min(multiprocessing.cpu_count() * 2, 4)
worker_class = "sync"  # Use "uvicorn.workers.UvicornWorker" for async

# Timeouts
timeout = 120  # Allow long-running ML operations
graceful_timeout = 30
keepalive = 5

# Logging
accesslog = "/var/log/svend/access.log"
errorlog = "/var/log/svend/error.log"
loglevel = "info"

# Process naming
proc_name = "svend"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
