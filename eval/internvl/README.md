# InternVL SpookyBench Evaluator

A command-line utility for evaluating different variants of InternVL models on the SpookyBench temporal pattern recognition dataset.

## Overview

This script allows you to evaluate various InternVL models on temporal pattern recognition tasks using the SpookyBench dataset. It supports different model variants including:

- InternVL2-40B
- InternVL2-8B
- InternVL2.5-78B
- InternVL2.5-8B
- InternVideo2-Chat-8B
- InternVL2.5-2B

The script is specifically designed for video analysis tasks, with a focus on recognizing temporal patterns in videos from the SpookyBench dataset.

## Prerequisites

- Python 3.10+
- PyTorch
- Transformers library
- Decord for video processing
- OpenCV (as fallback)

## Usage

The script supports evaluating videos from the SpookyBench dataset with different configuration options:

```bash
python run_internvl.py \
  --model internvl2.5-2b \
  --dataset /path/to/spookybench/directory \
  --csv /path/to/metadata.csv \
  --categories words shapes \
  --use_cot \
  --sample_size 10 \
  --output ./results
```

## Full Command-Line Arguments

```
usage: run_internvl.py [-h] [--model {internvl2-40b,internvl2-8b,internvl2.5-78b,internvl2.5-8b,internvideo2-chat-8b,internvl2.5-2b}] 
                      [--precision {fp16,bfloat16,fp32}] 
                      --dataset DATASET 
                      --csv CSV 
                      [--categories {words,shapes,images,videos} [{words,shapes,images,videos} ...]] 
                      [--use_cot] 
                      [--sample_size SAMPLE_SIZE] 
                      [--segments SEGMENTS] 
                      [--max_num MAX_NUM] 
                      [--input_size INPUT_SIZE] 
                      [--prompt PROMPT] 
                      [--output OUTPUT]

Run InternVL models on SpookyBench videos

optional arguments:
  -h, --help            show this help message and exit
  --model {internvl2-40b,internvl2-8b,internvl2.5-78b,internvl2.5-8b,internvideo2-chat-8b,internvl2.5-2b}
                        InternVL model variant to use
  --precision {fp16,bfloat16,fp32}
                        Model precision to use
  --dataset DATASET     Path to SpookyBench dataset directory or a single video file
  --csv CSV             Path to SpookyBench metadata CSV
  --categories {words,shapes,images,videos} [{words,shapes,images,videos} ...]
                        Categories to process (if not specified, all categories are used)
  --use_cot             Use chain-of-thought prompting instead of direct prompting
  --sample_size SAMPLE_SIZE
                        Number of videos to sample per category (if not specified, all videos are used)
  --segments SEGMENTS   Number of video segments to sample (default: 32)
  --max_num MAX_NUM     Maximum number of tiles per frame (default: 1)
  --input_size INPUT_SIZE
                        Size of input frames (default: 448)
  --prompt PROMPT       Custom prompt to use for all videos (overrides category-specific prompts)
  --output OUTPUT       Output directory for results (default: ./results)
```

## Examples

### Process Videos from a Specific Category

```bash
CUDA_VISIBLE_DEVICES=0 python run_internvl.py \
  --model internvl2.5-2b \
  --dataset /path/to/spookybench \
  --csv /path/to/metadata.csv \
  --categories words \
  --use_cot \
  --sample_size 5 \
  --output ./results
```

### Process Videos with Custom Parameters

```bash
CUDA_VISIBLE_DEVICES=0 python run_internvl.py \
  --model internvl2.5-2b \
  --dataset /path/to/spookybench \
  --csv /path/to/metadata.csv \
  --categories words shapes \
  --segments 48 \
  --input_size 448 \
  --max_num 1 \
  --output ./custom_results
```

### Use a Custom Prompt for All Videos

```bash
CUDA_VISIBLE_DEVICES=0 python run_internvl.py \
  --model internvl2.5-2b \
  --dataset /path/to/spookybench \
  --csv /path/to/metadata.csv \
  --prompt "What is the hidden message in this temporal pattern? Answer with just the message." \
  --output ./custom_prompt_results
```

## SpookyBench Categories

The SpookyBench dataset contains videos across several categories, each with specific temporal patterns:

1. **Words**: Text encoded through temporal patterns
2. **Shapes**: Geometric shapes encoded through temporal patterns
3. **Images**: Common objects encoded through temporal patterns
4. **Videos**: Movement encoded through temporal patterns

The script uses appropriate prompts for each category when processing the dataset, which can be enhanced with chain-of-thought (--use_cot) for better performance.

## Analyzing Results

After running the model on the SpookyBench dataset, you can analyze the results using the provided `analyze_results.py` script. This script compares model predictions against ground truth data and generates accuracy metrics and visualizations.

### Usage

```bash
python analyze_results.py \
  --results /path/to/results.json \
  --csv /path/to/metadata.csv \
  --output ./analysis_output
```

### Command-Line Arguments

```
usage: analyze_results.py [-h] --results RESULTS --csv CSV [--output OUTPUT]

Analyze InternVL SpookyBench results

optional arguments:
  -h, --help       show this help message and exit
  --results RESULTS  Path to JSON results file generated by run_internvl.py
  --csv CSV        Path to SpookyBench metadata CSV
  --output OUTPUT  Directory to save analysis results (default: ./analysis)
```

### Example Workflow

1. First, run the model on the SpookyBench dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python run_internvl.py \
  --model internvl2.5-2b \
  --dataset /path/to/spookybench \
  --csv /path/to/metadata.csv \
  --categories words \
  --use_cot \
  --output ./results
```

2. Then analyze the results:

```bash
python analyze_results.py \
  --results ./results/internvl2.5-2b_cot_[timestamp].json \
  --csv /path/to/metadata.csv \
  --output ./analysis_results
```

### Output

The analysis script generates:

1. A JSON file (`analysis.json`) containing detailed results for each video and category
2. A visualization (`accuracy.png`) showing accuracy metrics for each category and overall performance

This allows you to assess how well the model performs on different categories of temporal patterns in the SpookyBench dataset.

