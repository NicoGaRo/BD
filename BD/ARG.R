.
#Tweets SelArgentina
arg <- read.csv("a.csv", sep = ";", stringsAsFactors = FALSE)
arg <- arg[1:364,] #Se eliminan todas las observaciones vacias
arg$Sentiment <- factor(arg$Sentiment) #Conversion a factores de variable relevante
arg$Sentiment
str(arg)
library(tm)

#Limpieza datos
txt = arg$Tweet.Text
# remueve retweets
txtclean = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", txt)
# remueve @otragente
txtclean = gsub("@\\w+", "", txtclean)
# remueve simbolos de puntuación
txtclean = gsub("[[:punct:]]", "", txtclean)
# remove números
txtclean = gsub("[[:digit:]]", "", txtclean)
# remueve links
txtclean = gsub("http\\w+", "", txtclean)


arg_corpus <- VCorpus(VectorSource(txtclean))
#Creacion corpus de texto para analisis
print(arg_corpus)
arg_corpus_clean <- tm_map(arg_corpus, content_transformer(tolower))
#Transformacion de todos los tweets a minusculas, para eliminar posibles duplicados en el analisis

arg_corpus_clean <- tm_map(arg_corpus_clean, removeWords, c(stopwords("spanish"), "que", "un", "una", "por", "la", "el"))
#Remueve stopwords y palabras irrelevantes del corpus

sw <- readLines("C:/Users/NICOLAS GARZON/Downloads/Nueva carpeta (2)/BD/stopwords.es.txt",encoding="UTF-8")
sw = iconv(sw, to="ASCII//TRANSLIT")
#Archivo con nuevas stopwords en espanol

arg_corpus_clean <- tm_map(arg_corpus_clean, removeWords, sw) #Remueve stopwords faltantes
arg_corpus_clean <- tm_map(arg_corpus_clean, stripWhitespace) #Elimina espacios vacios resultantes
as.character(arg_corpus_clean[[3]])

#Tokenizacion
arg_dtm <- DocumentTermMatrix(arg_corpus_clean)
arg_dtm
#Conjuntos de entrenamiento y prueba
arg_dtm_train <- arg_dtm[1:218, ]
arg_dtm_test <- arg_dtm[219:364, ]

#Vectores de sentimiento
arg_train_labels <- arg[1:218, ]$Sentiment
arg_test_labels <- arg[219:364, ]$Sentiment

prop.table(table(arg_train_labels))
#Las proporciones entre las clases de tweets muestran una tendencia hacia la neutralidad (0.53), con la negatividad como
#segunda opcion (0.33) y los positivos solo (0.12)

prop.table(table(arg_test_labels))
#Para el conjunto de prueba se observa una distribucion similar (0.58 neutro, 0.27 negativo y 0.13 positivo)

wordcloud(arg_corpus_clean, min.freq=10, random.order=FALSE)
#Nube de palabras de corpus

pos <- subset(arg, Sentiment=="1")
neg <- subset(arg, Sentiment=="-1")
neu <- subset(arg, Sentiment=="0")
#Separacion de mensajes según su sentimiento

wordcloud(pos$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets positivos

wordcloud(neg$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets negativos

wordcloud(neu$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets neutros

#Terminos frecuentes
findFreqTerms(arg_dtm_train, 5)
arg_freq_words <- findFreqTerms(arg_dtm_train, 5)

#Eliminacion terminos irrelevantes o poco frecuentes del modelo
arg_dtm_freq_train <- arg_dtm_train[ , arg_freq_words]
arg_dtm_freq_test <- arg_dtm_test[ , arg_freq_words]

#Funcion para indicar si los tweets contienen o no terminos frecuentes
convert_counts <- function(x){
  x<- ifelse(x>0, "Yes", "No")
}

arg_train <- apply(arg_dtm_freq_train, MARGIN=2, convert_counts)
arg_test <- apply(arg_dtm_freq_test, MARGIN=2, convert_counts)


#Modelo de prediccion
arg_classifier <- naiveBayes(arg_train, arg_train_labels)
arg_test_pred <- predict(arg_classifier, arg_test)
CrossTable(arg_test_pred, arg_test_labels, prop.chisq=FALSE, prop.t=FALSE, dnn=c('Prediccion', 'Real'))

#El desempeño del modelo es bastante regular, ya que de los 40 casos de tweets positivos, predice 56; de los 86 tweets neutrales
#predice solo 49 y de los 20 tweets positivos predice 41.

#Modelo 2
arg_classifier2 <- naiveBayes(arg_train, arg_train_labels, laplace=1)
arg_test_pred2 <- predict(arg_classifier2, arg_test)
CrossTable(arg_test_pred2, arg_test_labels, prop.chisq=FALSE, prop.t=FALSE, dnn=c('Prediccion', 'Real'))

#Las predicciones de este modelo empeoran los resultados, ya que predice 94 tweets negativos, 51 neutrales y tan solo 1 positivo.
#Evidentemente ambos modelos tienden a aumentar las predicciones negativas dada la ambiguedad
#del lenguaje de los tweets neutrales; sin embargo es claro que aun no se cuenta con un modelo confiable para la prediccion del sentimiento
#de los twiteros.
