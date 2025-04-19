import argparse
import logging
from datetime import datetime
import sys
import os

from classes.model_adapter.model_adapter_factory import ModelAdapterFactory
from utils.error_handler import ResearchAgentError, handle_exceptions
from utils.logging_config import configure_logging
from utils.logger import Logger
from classes.model_adapter.claude_model_adapter import ClaudeModelAdapter
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
    parser.add_argument('--focus', type=str, default='', help='Additional summary focus')
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )

    args = parser.parse_args()

    # Additional validation for date range
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("Start date must be before end date.")

    # Ensure the query is not empty after stripping whitespace
    if not args.query.strip():
        parser.error("Research query cannot be empty.")

    return args

@handle_exceptions(error_type=ResearchAgentError)
def main():
    """Main entry point for the research agent."""
    args = parse_arguments()
    log_level = getattr(logging, args.log_level)
    logger = configure_logging(log_level=log_level)

    # Print the validated input
    Logger.info(logger,"\n=== Research Agent Parameters ===")
    Logger.info(logger,f"Query: {args.query}")
    Logger.info(logger,f"Date Range: {args.start_date or 'Not specified'} to {args.end_date or 'Not specified'}")
    Logger.info(logger,f"Number of Papers: {args.paper_count}")
    Logger.info(logger,f"Focus Area: {args.focus}")
    Logger.info(logger,"================================\n")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        Logger.info(logger,"ANTHROPIC_API_KEY undefined! Please set it in your environment variables.")
        return 1
    model_adapter = ModelAdapterFactory.create_adapter("claude")
    research_agent = ResearchAgent(args, model_adapter)
    research_agent.research_pipeline()

    Logger.info(logger,"Research agent complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())