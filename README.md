# CAND-BERT

## File Structure
```markdown
├── bert-base-chinese
├── data
├── logs
├── README.md
├── requirements.txt
├── result
├── saved_models
└── src
```

## Setup
`pip install -r requirements`

## Train & Eval
> Hypyer Parameters' Setting is in src/param_config.py
```shell
cd src
python run.py
```

## Inference
```shell
python predict.py
```