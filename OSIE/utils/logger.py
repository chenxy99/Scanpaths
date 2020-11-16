import logging

def Logger(log_file):
    # initialize logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create a handler to write to the file
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)

    # create a handler to write to the command
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # format definition
    formatter = logging.Formatter('[%(asctime)s - %(name)s - %(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
