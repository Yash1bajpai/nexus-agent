# Demo buggy code for testing Nexus-Agent debug capabilities
# Throws ZeroDivisionError when processing an empty list of metrics

def calculate_average_response_time(response_times: list[float]) -> float:
    """Calculate average response time across server logs."""
    total_time = 0.0
    for time_ms in response_times:
        total_time += time_ms
    
    # Bug: Division by zero when response_times list is empty
    return total_time / len(response_times)


if __name__ == "__main__":
    # Simulate processing empty batch of server logs
    empty_logs = []
    avg = calculate_average_response_time(empty_logs)
    print(f"Average response time: {avg:.2f}ms")
