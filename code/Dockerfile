FROM python:3.11-slim

WORKDIR /educeq

COPY requirements.lock pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.lock

COPY . .

# The test suite, then every table of Section 5.
CMD ["make", "all"]
