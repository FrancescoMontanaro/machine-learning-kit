def progress_bar(iteration: int, total: int, prefix: str = 'Progress', suffix: str = 'Complete', decimals: int = 1, length: int = 50, fill: str = '█', print_end: str = "\r") -> None:
    """
    Function that prints a progress bar
    
    Parameters:
    - iteration (int): The current iteration
    - total (int): The total number of iterations
    - prefix (str): The prefix of the progress bar
    - suffix (str): The suffix of the progress bar
    - decimals (int): The number of decimals
    - length (int): The length of the progress bar
    - fill (str): The fill character
    - printEnd (str): The print end character
    """
    
    # Computing the percentage of completion
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    
    # Printing the percentage
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = print_end)
    if iteration == total: print()