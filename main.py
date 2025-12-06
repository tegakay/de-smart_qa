from smart_qa.client import LLMClient
import logging
import argparse
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="Utils INFO: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Simple LLM Client")

    parser.add_argument(
        "--file",
        type=str,
        default='file.txt',
        help="Load files from a speciified path",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save conversation history or summary to a file",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clears the LLM client's cache",
    )


    llm_output = ''

    args = parser.parse_args()
    client = LLMClient()
    

    if args.clear_cache:
        LLMClient.cached_summarize.cache_clear()

    if args.file:
     text =   client.read_text_file(args.file)
     logger.info(text)
     summary = client.summarize(text)
     print(summary)
     llm_output = summary
    
    if args.save:
        filename = os.path.join(os.getcwd(), 'history.txt')
        try:
            with open(filename, mode='a') as f:
                f.write(text)
                f.write('\n')
                f.write(llm_output)

        except Exception as e:
            print(f"Error writing to file: {e}")

    return 


    # boy = client.summarize("The term random string in the context of summarization generally refers to meaningless placeholder text used for design or testing purposes that is intended not to distract from the layout or functionality being evaluated. It can also refer to the technical process of generating unique strings for use in programming or security applications")
    # print(boy)
    # logger.info("Summarization complete.")
    # boy2 = client.summarize("The term random string in the context of summarization generally refers to meaningless placeholder text used for design or testing purposes that is intended not to distract from the layout or functionality being evaluated. It can also refer to the technical process of generating unique strings for use in programming or security applications")
    # print(boy2)


if __name__ == "__main__":
    main()