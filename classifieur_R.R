classifieur <- function(dataset) {
  # Chargement des données construites lors de l'apprentissage 
  require(MASS)
  require(caret)
  load("env_R.Rdata",.GlobalEnv)
  predictions <- predict.train(.GlobalEnv$classifieur, dataset[,-31])
  return(predictions)
}

