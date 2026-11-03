"""Engineering bounds for physical text inspection, independent of history retention."""

# Conservative local-text workload bound, independent of history artifacts.
# Native spawned-worker benchmarks cover dense lines, Unicode and a 1024-byte
# prefix-stress query at/beyond this ceiling; it is not a semantic/permission limit.
MAX_SCAN_BYTES = 256 * 1024
READ_CHUNK_BYTES = 4096

# Bounds KMP prefix/window state to O(1024) codepoints, not a line/file buffer.
MAX_LITERAL_BYTES = 1024

# Serialized private requests carry exact arguments; cwd is inherited, not reopened.
# Responses echo arguments twice (envelope and receipt), plus bounded fixed fields.
MAX_REQUEST_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 2 * MAX_REQUEST_BYTES + 2048
