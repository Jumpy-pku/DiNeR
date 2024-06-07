# DiNeR (DIsh NamE Recognition)
Source code and data for [DiNeR: a Large Realistic Dataset for Evaluating Compositional Generalization](https://aclanthology.org/2023.emnlp-main.924/) (EMNLP 2023 main conference paper)

You can try our model on the [Hugging Face Model Space](https://huggingface.co/spaces/Jumpy-pku/dish-name-recognition)

## Dependencies

- torch == 2.2.0
- transformers == 4.33.2


## Data

Our data is in the `data/` folder. 

- `tmcd_data.json`: The training, valid and TMCD test set for the dish name recognition task.
- `glossary.json`: The glossary of food, actions and flavours. In each dicionary, the key is the name of the food, action or flavour, and the value is the corresponding cluster ID. (We discuss the glossary clustering in section 3.2 in the paper.)

## Training and Evaluation

To train and evaluate the plain seq2seq method, run the following command:

```bash
# fine-tuning from plain T5
python main.py --task name --model_path Langboat/mengzi-t5-base
# fine-tuning from continue pre-trained T5
python main.py --task name --model_path Jumpy-pku/t5-recipe-continue-pretrained
```

To train and evaluate the proposed CP-FT method, fisrt fine-tune the auxiliary model by running the following command:

```bash
# fine-tuning from plain T5
python main.py --task component --model_path Langboat/mengzi-t5-base --eval_step 3000
# fine-tuning from continue pre-trained T5
python main.py --task component --model_path Jumpy-pku/t5-recipe-continue-pretrained --eval_step 3000
```

Then fine-tune the main model by running the following command:

```bash
# fine-tuning from plain T5
python main.py --task component --model_path Langboat/mengzi-t5-base --pred_path outputs/component_42_mengzi-t5-base/preds.pt --epochs 3
# fine-tuning from continue pre-trained T5
python main.py --task component --model_path Jumpy-pku/t5-recipe-continue-pretrained --pred_path outputs/component_42_t5-recipe-continue-pretrained/preds.pt --epochs 3
```

## Citation

Please cite our paper if this repository inspires your work.
```
@inproceedings{hu-etal-2023-diner,
    title = "{D}i{N}e{R}: A Large Realistic Dataset for Evaluating Compositional Generalization",
    author = "Hu, Chengang  and
      Liu, Xiao  and
      Feng, Yansong",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.924",
    doi = "10.18653/v1/2023.emnlp-main.924",
    pages = "14938--14947",
}
```

## Contact

If you have any questions regarding the code, please create an issue or contact the owner of this repository.
