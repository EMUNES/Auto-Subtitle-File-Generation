# TODO

This project needs many more improvement to make it better!

## Problems to solve

### can't accurately recognize the ending and beginning of speech

- Add more clips to train.
- Try to build dataset with smaller clip length.

### can't break the sentense properly

- Implement hooks to recognize the long clips that needs break and get timestamp for breakpoint. - IN DEVELOPMENT

### can be especially disrupted by music and animal sound that's near speech

- Increase the threshold.
- Find more clips about animal sound when building the dataset.

## Other improvements

- Use a lighter model, such as resNest, denseNet and mobileNet.
- Ensemble models.

## future development

ASG should be able to take your translations and break them into slices and put them into the subtitle file it generates. This function is very useful and will be the next goal after current functionalities are stable and effective.

## And may more...
