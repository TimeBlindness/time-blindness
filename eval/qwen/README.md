# Qwen VL SpookyBench Evaluator

A command-line utility for evaluating different variants of Qwen VL models on the SpookyBench temporal pattern recognition dataset.

## Overview

This script allows you to evaluate various Qwen VL models on temporal pattern recognition tasks using the SpookyBench dataset. It supports different model variants including:

- Qwen2-VL-2B-Instruct
- Qwen2-VL-7B-Instruct
- Qwen2-VL-72B-Instruct
- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-72B-Instruct

The script is specifically designed for video analysis tasks, with a focus on recognizing temporal patterns in videos from the SpookyBench dataset.

## Prerequisites

- Python 3.10+
- PyTorch
- Transformers library
- OpenCV for video processing
- Pandas for dataset handling

## Usage

The script supports evaluating videos from the SpookyBench dataset with different configuration options:

```bash
python run_qwen.py \
  --model qwen2-vl-7b \
  --dataset /path/to/spookybench/directory \
  --csv /path/to/metadata.csv \
  --categories words shapes \
  --use_cot \
  --sample_size 10 \
  --output ./results
```

## Full Command-Line Arguments

```
usage: run_qwen.py [-h] [--model {qwen2-vl-2b,qwen2-vl-7b,qwen2-vl-72b,qwen2.5-vl-3b,qwen2.5-vl-7b,qwen2.5-vl-72b}] 
                   [--precision {fp16,bfloat16,fp32}] 
                   --dataset DATASET 
                   --csv CSV 
                   [--categories {words,shapes,images,videos} [{words,shapes,images,videos} ...]] 
                   [--use_cot] 
                   [--sample_size SAMPLE_SIZE] 
                   [--fps FPS] 
                   [--max_pixels MAX_PIXELS] 
                   [--prompt PROMPT] 
                   [--output OUTPUT]

Run Qwen VL models on SpookyBench videos

optional arguments:
  -h, --help            show this help message and exit
  --model {qwen2-vl-2b,qwen2-vl-7b,qwen2-vl-72b,qwen2.5-vl-3b,qwen2.5-vl-7b,qwen2.5-vl-72b}
                        Qwen VL model variant to use
  --precision {fp16,bfloat16,fp32}
                        Model precision to use
  --dataset DATASET     Path to SpookyBench dataset directory or a single video file
  --csv CSV             Path to SpookyBench metadata CSV
  --categories {words,shapes,images,videos} [{words,shapes,images,videos} ...]
                        Categories to process (if not specified, all categories are used)
  --use_cot             Use chain-of-thought prompting instead of direct prompting
  --sample_size SAMPLE_SIZE
                        Number of videos to sample per category (if not specified, all videos are used)
  --fps FPS             Frames per second to sample from video (default: 30.0)
  --max_pixels MAX_PIXELS
                        Maximum number of pixels per frame (default: 202500)
  --prompt PROMPT       Custom prompt to use for all videos (overrides category-specific prompts)
  --output OUTPUT       Output directory for results (default: ./results)
```

## Examples

### Process Videos from a Specific Category

```bash
CUDA_VISIBLE_DEVICES=0 python run_qwen.py \
  --model qwen2-vl-7b \
  --dataset /path/to/spookybench \
  --csv /path/to/metadata.csv \
  --categories words \
  --use_cot \
  --sample_size 5 \
  --output ./results
```

### Process Videos with Custom Parameters

```bash
CUDA_VISIBLE_DEVICES=0 python run_qwen.py \
  --model qwen2.5-vl-7b \
  --dataset /path/to/spookybench \
  --csv /path/to/metadata.csv \
  --categories words shapes \
  --fps 60.0 \
  --max_pixels 360000 \
  --output ./custom_results
```

### Use a Custom Prompt for All Videos

```bash
CUDA_VISIBLE_DEVICES=0 python run_qwen.py \
  --model qwen2.5-vl-3b \
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
We also did human in the loop evaluation since the datset size is small. For smaller dataset we recommend reviewing the model's response manually as well.
### Command-Line Arguments

```
usage: analyze_results.py [-h] --results RESULTS --csv CSV [--output OUTPUT]

Analyze Qwen VL SpookyBench results

optional arguments:
  -h, --help       show this help message and exit
  --results RESULTS  Path to JSON results file generated by run_qwen.py
  --csv CSV        Path to SpookyBench metadata CSV
  --output OUTPUT  Directory to save analysis results (default: ./analysis)
```

### Example Workflow

1. First, run the model on the SpookyBench dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python run_qwen.py \
  --model qwen2-vl-7b \
  --dataset /path/to/spookybench \
  --csv /path/to/metadata.csv \
  --categories words \
  --use_cot \
  --output ./results
```

2. Then analyze the results:

```bash
python analyze_results.py \
  --results ./results/qwen2-vl-7b_cot_[timestamp].json \
  --csv /path/to/metadata.csv \
  --output ./analysis_results
```

### Output

The analysis script generates:

1. A JSON file (`analysis.json`) containing detailed results for each video and category
2. A visualization (`accuracy.png`) showing accuracy metrics for each category and overall performance

This allows you to assess how well the model performs on different categories of temporal patterns in the SpookyBench dataset.
