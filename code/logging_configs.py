import logging

def setup_logging_simulation(path):
    # Configure logging with timestamps
    logging.basicConfig(
        filename=path,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Return the root logger
    return logging.getLogger()