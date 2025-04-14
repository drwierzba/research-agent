import argparse
import re
from datetime import datetime
import sys
import getpass
import os

from services.langchain import ResearchAgent


def validate_date(date_str):
    """Validate date string format (YYYY-MM-DD)."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Please use YYYY-MM-DD.")


def validate_positive_int(value):
    """Validate that the input is a positive integer."""
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description='LLM-Based Research Agent for Literature Review')

    # Required argument: research query/topic
    parser.add_argument('query', type=str,
                        help='Research query or topic (e.g., "Graph neural networks for traffic prediction")')

    # Optional parameters
    parser.add_argument('--start-date', type=validate_date, help='Start date for paper search (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=validate_date, help='End date for paper search (format: YYYY-MM-DD)')
    parser.add_argument('--paper-count', type=validate_positive_int, default=20,
                        help='Number of papers to retrieve (default: 20)')
    parser.add_argument('--focus', type=str, choices=['methodology', 'datasets', 'performance', 'all'],
                        default='all', help='Focus area of the research (default: all)')

    args = parser.parse_args()

    # Additional validation for date range
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("Start date must be before end date.")

    # Ensure the query is not empty after stripping whitespace
    if not args.query.strip():
        parser.error("Research query cannot be empty.")

    return args


def main():
    """Main entry point for the research agent."""
    try:
        args = parse_arguments()

        # Print the validated input
        print("\n=== Research Agent Parameters ===")
        print(f"Query: {args.query}")
        print(f"Date Range: {args.start_date or 'Not specified'} to {args.end_date or 'Not specified'}")
        print(f"Number of Papers: {args.paper_count}")
        print(f"Focus Area: {args.focus}")
        print("================================\n")

        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY undefined! Please set it in your environment variables.")
            return 1
        research_agent = ResearchAgent(args)
        research_agent.research_pipeline()

        print("Research agent complete.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
