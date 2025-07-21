#!/usr/bin/env python3
# ascii_banner.py - Displays a colorful "Conviction AI" ASCII art banner

import random
from colorama import init, Fore, Style

# Initialize colorama
init()

def print_conviction_ai_banner():
    # ASCII art for "Conviction AI"
    banner = r"""
     ██████╗ ██████╗ ███╗   ██╗██╗   ██╗██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
    ██╔════╝██╔═══██╗████╗  ██║██║   ██║██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
    ██║     ██║   ██║██╔██╗ ██║██║   ██║██║██║        ██║   ██║██║   ██║██╔██╗ ██║
    ██║     ██║   ██║██║╚██╗██║╚██╗ ██╔╝██║██║        ██║   ██║██║   ██║██║╚██╗██║
    ╚██████╗╚██████╔╝██║ ╚████║ ╚████╔╝ ██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
     ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝  ╚═══╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                                                                                   
             █████╗ ██╗
            ██╔══██╗██║
            ███████║██║
            ██╔══██║██║
            ██║  ██║██║
            ╚═╝  ╚═╝╚═╝
    """
    
    # Different color schemes (primary, secondary)
    color_schemes = [
        (Fore.BLUE, Fore.CYAN),       # Blue & Cyan
        (Fore.GREEN, Fore.YELLOW),    # Green & Yellow
        (Fore.MAGENTA, Fore.CYAN),    # Magenta & Cyan
        (Fore.RED, Fore.YELLOW),      # Red & Yellow
        (Fore.CYAN, Fore.GREEN),      # Cyan & Green
        (Fore.YELLOW, Fore.RED)       # Yellow & Red
    ]
    
    # Randomly choose a color scheme
    primary_color, secondary_color = random.choice(color_schemes)
    
    # Print the banner with alternating colors for each line
    lines = banner.split('\n')
    for i, line in enumerate(lines):
        if i % 2 == 0:
            print(f"{primary_color}{line}{Style.RESET_ALL}")
        else:
            print(f"{secondary_color}{line}{Style.RESET_ALL}")
    
    # Print a tagline with a different style
    taglines = [
        "Predictive Intelligence for Financial Markets",
        "AI-Powered Investment Decisions",
        "Algorithmic Conviction for Smarter Trading",
        "Data-Driven Investment Strategies",
        "ML Models with Market Conviction"
    ]
    
    print(f"\n{Style.BRIGHT}{Fore.WHITE}▶ {random.choice(taglines)}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    print_conviction_ai_banner()
