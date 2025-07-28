from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from jsonschema import validate
from dotenv import load_dotenv
from tqdm.auto import tqdm
import multiprocessing
import argparse
import json
import sys
import os
import re


def load_llm(model_name, **kwargs):

    name_parts = model_name.split(':')
    assert len(name_parts) == 2

    base_name = name_parts[0]
    version_name = name_parts[1]

    if base_name == 'openai':
        llm = ChatOpenAI(
            model       = version_name,
            temperature = 1, # OpenAI complains about temperature different from 1
            max_tokens  = 4096,
            max_retries = 5,
            **kwargs
        )
    elif base_name == 'anthropic':
        llm = ChatAnthropic(
            model       = version_name,
            temperature = 0,
            max_tokens  = 4096,
            max_retries = 5,
            **kwargs
        )
    elif base_name == 'google':
        llm = ChatGoogleGenerativeAI(
            model       = version_name,
            temperature = 0,
            max_tokens  = 16384,
            max_retries = 5,
            **kwargs
        )
    elif base_name == 'nomodel':
        llm = None
    else:
        llm = ChatOllama(model=model_name, **kwargs)
        llm.num_ctx     = 16384
        llm.num_predict = 4096
        llm.temperature = 0

    return llm


def get_next_nonspace(text, start_pos):
    for i in range(start_pos, len(text)):
        if not text[i].isspace():
            return text[i], i
    return None, -1


def valid_quote_json(text, quote_pos):

    next_char, next_pos = get_next_nonspace(text, quote_pos+1)
    valid = True
    next_string_found = False

    while next_char is not None and not next_string_found and valid:

        next_next_char, next_next_pos = get_next_nonspace(text, next_pos+1)

        if next_char in ['}', ']']:
            valid = next_next_char in [',', '}', ']'] or next_next_char is None
        elif next_char == ':':
            next_string_found = next_next_char.isdigit() or next_next_char == '"'
            valid = next_next_char in ['{', '['] or next_string_found
        elif next_char == ',':
            next_string_found = next_next_char == '"'
            valid = next_string_found
        else:
            valid = False

        next_char, next_pos = next_next_char, next_next_pos

    return valid


def fix_json(raw_data):

    inside_string = False
    escaped_chars_map = {'\n': '\\n', '\t': '\\t'}
    res = ""
    prev_char = None

    i = 0
    while i < len(raw_data):

        next_str = raw_data[i]

        if raw_data[i] == '"':
            match_noquote_value = re.match('^[\\s]*:[\\s]*[^"\\s\\d]', raw_data[i+1:])

            if not inside_string:
                inside_string = True
            elif match_noquote_value is not None:
                next_str = '": "'
                inside_string = True
                i += match_noquote_value.end() - 1
            elif valid_quote_json(raw_data, i):
                inside_string = False
            elif prev_char != '\\':
                next_str = '\\"'

            # Sometimes llama escapes a end string quote, so unescape it
            if not inside_string and prev_char == '\\':
                res = res[:-1]
        elif inside_string and raw_data[i] in escaped_chars_map:
            next_str = escaped_chars_map[raw_data[i]]

        res = res + next_str
        prev_char = raw_data[i]
        i += 1

    return res


def parse_json(raw_data):
    """
    Tries to parse a string as a JSON object

    Parameters:
      raw_data (str): String containing JSON data

    Returns:
      data (json or NoneType): JSON object if a valid JSON is encoded into the string. Otherwise None is returned
    """

    fixed_raw_data = fix_json(raw_data)

    try:
        data = json.loads(fixed_raw_data)
    except json.decoder.JSONDecodeError as e:
        # print(f'WARNING: could not parse list from answer ({raw_data})')
        data = None

    return data


def valid_schema(json_object, schema):
    """
    Validates a JSON object such that it follows a specific schema. It checks for several fields to be present, but extra fields are allowed

    Parameters:
      json_object (json): JSON to be validated
      schema (dict): Schema used to validate the JSON object. Follow the syntax from "https://python-jsonschema.readthedocs.io/en/latest/validate/"

    Returns:
      res (bool): Returns True if the JSON follows the schema. False otherwise
    """

    try:
        validate(instance=json_object, schema=schema)
        return True
    except:
        return False


def extract_all_json(raw_data, schema=None, sort_by_length=True):
    """
    Extract all valid JSONs in a string. There might be extra characters between JSON objects. If a schema is provided,
    JSON objects are filtered so that they follow the provided schema

    Parameters:
      raw_data (str): String containing JSON objects
      schema (dict or NoneType): Schema used to filter extracted JSON objects. If not provided, the filter will not be applied

    Returns:
      res (list[json]): List of all valid JSON objects encoded into the string
    """

    res = []
    lengths = []
    start_pos = 0
    reading_json = False
    reading_list = False
    depth = 0
    for i,c in enumerate(raw_data):
        if not reading_json and not reading_list and c == '{':
            reading_json = True
            depth = 0
            start_pos = i
        if not reading_json and not reading_list and c == '[':
            reading_list = True
            depth = 0
            start_pos = i
        if reading_json and c == '{':
            depth += 1
        if reading_list and c == '[':
            depth += 1
        if reading_json and c == '}' and depth > 0:
            depth -= 1
        if reading_list and c == ']' and depth > 0:
            depth -= 1
        if (reading_json and c == '}' and depth == 0) or (reading_list and c == ']' and depth == 0):
            reading_json = False
            reading_list = False
            json_data = parse_json(raw_data[start_pos:i+1])
            if json_data is not None:
                res.append(json_data)
                lengths.append(i - start_pos)

    if sort_by_length:
        res = sorted(zip(res, lengths), key=lambda elem: -elem[1])
        res = [elem[0] for elem in res]

    if schema is not None:
        res = list(filter(lambda json_object: valid_schema(json_object, schema), res))

    return res


def parse_response(response, num_choices):
    schema = {
        "type": "object",
        "properties": {
            "choice_position": {"type": ["string", "integer"]},
            "choice_string": {"type": "string"},
        },
        "required": ["choice_position", "choice_string"]
    }

    reason = None
        
    parsed_response = extract_all_json(response, schema)
    if len(parsed_response) == 0:
        parsed_response = None
        reason = 'Could no parse any JSON in the response'

    if parsed_response is not None:
        parsed_response = parsed_response[0]

    if parsed_response is not None and isinstance(parsed_response['choice_position'], str):
        try:
            parsed_response['choice_position'] = int(parsed_response['choice_position'])
        except:
            parsed_response = None
            reason = 'Could not parse answer position'

    if parsed_response is not None:
        if parsed_response['choice_position'] < 1 or parsed_response['choice_position'] > num_choices:
            parsed_response = None
            reason = f'Answer position not in the range [1, {num_choices}]'
        else:
            parsed_response['choice_position'] = parsed_response['choice_position'] - 1

    return parsed_response, reason


def process_benchmark(benchmark, llm_name, llm_host, results, results_queue, process_id, num_processes):

    # Load recommender LLM
    llm = load_llm(llm_name, base_url=llm_host)

    # Round-robin parallelization
    for sample_pos in range(process_id, len(benchmark['samples']), num_processes):

        if results[sample_pos] is not None:
            sample_info = results[sample_pos]
        else:
            # Generate response
            prompt = benchmark['samples'][sample_pos]['prompt']
            prompt = prompt.replace('{', '{{')
            prompt = prompt.replace('}', '}}')
            prompt = ChatPromptTemplate.from_template(prompt)
            chain = prompt | llm
            response = chain.invoke(dict()).text()

            # Parse response
            num_choices = len(benchmark['samples'][sample_pos]['data']['choices'])
            parsed_response, reason = parse_response(response, num_choices)

            # Store results
            sample_info = {
                'predicted_pos': parsed_response['choice_position'] if parsed_response is not None else None,
                'reason': reason,
                'response': response,
                'parsed_response': parsed_response
            }

        results_queue.put((sample_info, sample_pos))

def distribute_work(input_benchmark_path, llm_name, hosts, output_path, store_interval):

    # Load evaluation dataset
    with open(input_benchmark_path, 'r') as f:
        benchmark = json.load(f)

    # Prepare results structure
    results = {
        'benchmark': input_benchmark_path,
        'llm': llm_name
    }

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            old_results = json.load(f)
        for key in results.keys():
            assert results[key] == old_results[key]
        results = old_results
    else:
        results['results'] = [None] * len(benchmark['samples'])

    # Split dataset and launch processes
    results_queue = multiprocessing.Queue()
    processes = []
    for i, llm_host in enumerate(hosts):
        process = multiprocessing.Process(target=process_benchmark, args=(benchmark, llm_name, llm_host, results['results'], results_queue, i, len(hosts)))
        process.start()
        processes.append(process)

    # Collect all results
    for i in tqdm(range(len(benchmark['samples'])), desc='Evaluation'):
        sample_info, sample_pos = results_queue.get()
        results['results'][sample_pos] = sample_info

        # Store generated results
        if i % store_interval == 0 or i+1 == len(benchmark['samples']):
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=3)

    # Wait for all processes to finish (they should be already finished)
    for process in processes:
        process.join()

def main(args):

    load_dotenv()

    input_benchmark_path = args.input_benchmark_path
    llm_name = args.llm_name
    output_path = args.output_path
    hosts_path = args.hosts_path
    store_interval = args.store_interval

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configure LLM recommenders
    if hosts_path is None:
        hosts = [None]
    else:
        with open(hosts_path, 'r') as f:
            hosts = list(map(lambda line: line.strip(), f.readlines()))

    # Launch processes
    distribute_work(input_benchmark_path, llm_name, hosts, output_path, store_interval)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate responses from a benchmark')
    parser.add_argument('-i', '--input-benchmark-path', required=True, type=str, help='Benchmark path to be evaluated')
    parser.add_argument('-m', '--llm_name', required=True, type=str, help='Evaluated LLM name. Use Ollama format. If results path is included')
    parser.add_argument('-o', '--output-path', required=True, type=str, help='Path to JSON-like file where results will be stored')
    parser.add_argument('--hosts-path', type=str, default=None, help='Path to a file of hosts used to do inference with Ollama models. The number of hosts will create the same number of parallel processes. If not provided, the default host will be used')
    parser.add_argument('--store-interval', type=int, default=1, help='Interval used to store the data for recovery purposes')

    args = parser.parse_args()

    main(args)
