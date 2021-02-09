# TODO

This project need more improvement to make it better!

## Problems to solve

### can't accurately recognize the ending and beginning of speech

- inference with smaller clip duration
- increase the threshold

### can't break the sentense properly

- inference with smaller clip duration
- increase the threshold
- add a dedicated hook to handle the result if former are not working

### can be especially disrupted by music and animal sound that's near speech

- inference with smaller clip duration
- increase the threshold

### the model is always overfitting

- change a lighter model. Candidates: dense121 and mobilenet v2.

### training time too long locally

- use fp16 for training. Try solving fp16 NaN problem by add normalization in model or prevent NaN or Infinite values by using other methods.

## Possibles

- change to a binary classification problem
- dialogues vary much in length, this may cause data imbalance
