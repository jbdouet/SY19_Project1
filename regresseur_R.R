regresseur <- function(dataset) {
  load("env_R.Rdata", .GlobalEnv)
  require(MASS)
  require(caret)
  predictions <- caret::predict.train(.GlobalEnv$regresseur, dataset[,-51])
  return(predictions)
}


