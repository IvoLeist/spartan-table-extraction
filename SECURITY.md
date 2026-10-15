# Security Policy

Please do not report security problems in public issues. Use GitHub's
**Report a vulnerability** button on the repository's *Security* tab instead, so the problem
can be fixed before it is disclosed.

Security-relevant areas include the REST API (`spartan/api.py`: upload handling, size limits,
CORS) and the handling of credentials for optional cloud services, which must only ever come
from environment variables.
