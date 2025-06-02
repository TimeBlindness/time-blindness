# Closed Model Evaluation for SpookyBench

This directory contains scripts for evaluating SpookyBench videos using closed (API-based) multimodal models:

1. **GPT-4o** (`eval_gpt4o.py`): Evaluate videos using OpenAI's GPT-4o or GPT-4o-mini models
2. **Gemini** (`eval_gemini.py`): Evaluate videos using Google's Gemini 1.5 Pro or Gemini 2.0 Flash models

These scripts follow the same structure and interface as the open-source model evaluations (MovieChat, Qwen) to allow for easy comparison.

## Setup

### Prerequisites

- Python 3.10+
- API keys for OpenAI and/or Google Gemini
- SpookyBench dataset videos
- SpookyBench metadata CSV file

### Installation

1. Install the required dependencies:

```bash
pip install openai google-generativeai pandas tqdm python-dotenv
```

2. Set up your API keys in the `.env` file:

```
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

## Usage

### GPT-4o Evaluation

```bash
python eval_gpt4o.py \
  --dataset /path/to/SpookyBenchDatasets \
  --csv /path/to/metadata.csv \
  --output ./gpt4o_results \
  --model gpt-4o \
  --categories words shapes \
  --use_cot \
  --sample_size 5
```

### Gemini Evaluation

```bash
python eval_gemini.py \
  --dataset /path/to/SpookyBenchDatasets \
  --csv /path/to/metadata.csv \
  --output ./gemini_results \
  --model gemini-1.5-pro \
  --categories words shapes \
  --use_cot \
  --sample_size 5
```

## Command-Line Arguments

Both scripts accept the same command-line arguments:

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--dataset` | Path to SpookyBench dataset directory | Yes | - |
| `--csv` | Path to SpookyBench metadata CSV | Yes | - |
| `--output` | Output directory for results | Yes | - |
| `--model` | Model to use (depends on script) | No | gpt-4o / gemini-1.5-pro |
| `--categories` | Categories to process | No | All categories |
| `--use_cot` | Use chain-of-thought prompting | No | False |
| `--sample_size` | Number of videos per category | No | All videos |

### Model Options

- **GPT-4o**: `gpt-4o`, `gpt-4o-mini`
- **Gemini**: `gemini-1.5-pro`, `gemini-2.0-flash`

## Output

Results are saved as JSON files in the specified output directory with the following naming convention:

```
{model_name}_results_{timestamp}.json
```

The JSON file contains key-value pairs where:
- Keys are video paths relative to the dataset directory (e.g., "words/abcd1234.mp4")
- Values are the model's responses for each video

## Processing Details

- Videos are processed by category, with each video uploaded to the respective API
- Both scripts use a set of category-specific prompts for direct and chain-of-thought prompting
- Error handling and retry mechanisms are implemented for API failures
- Memory cleanup and waiting periods are included between processing videos

## Notes

- We also perform human-in-the-loop evaluation for SpookyBench since the dataset size is small, allowing manual verification of responses
- For videos where the models fail to process or reach the API limits, the responses will contain error messages
- The scripts include rate limiting protections with wait times between requests

## License

This project is licensed under the same terms as the SpookyBench dataset. Please refer to the dataset's documentation for more details.
