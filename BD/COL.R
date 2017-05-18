.
#Tweets SelColombia
col <- read.csv("co.csv", sep = ";", stringsAsFactors = FALSE)
col <- col[1:462,] #Se eliminan todas las observaciones vacias
col$Sentiment <- factor(col$Sentiment) #Conversion a factores de variable relevante
col$Sentiment
str(col)
library(tm)

#Limpieza datos
txt = col$Tweet.Text
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


col_corpus <- VCorpus(VectorSource(txtclean))
#Creacion corpus de texto para analisis
print(col_corpus)
col_corpus_clean <- tm_map(col_corpus, content_transformer(tolower))
#Transformacion de todos los tweets a minusculas, para eliminar posibles duplicados en el analisis

col_corpus_clean <- tm_map(col_corpus_clean, removeWords, c(stopwords("spanish"), "que", "un", "una", "por", "la", "el"))
#Remueve stopwords y palabras irrelevantes del corpus

sw <- readLines("C:/Users/NICOLAS GARZON/Downloads/Nueva carpeta (2)/BD/stopwords.es.txt",encoding="UTF-8")
sw = iconv(sw, to="ASCII//TRANSLIT")
#Archivo con nuevas stopwords en espanol

col_corpus_clean <- tm_map(col_corpus_clean, removeWords, sw) #Remueve stopwords faltantes
col_corpus_clean <- tm_map(col_corpus_clean, stripWhitespace) #Elimina espacios vacios resultantes
as.character(col_corpus_clean[[3]])

#Tokenizacion
col_dtm <- DocumentTermMatrix(col_corpus_clean)
col_dtm
#Conjuntos de entrenamiento y prueba
col_dtm_train <- col_dtm[1:277, ]
col_dtm_test <- col_dtm[278:462, ]

#Vectores de sentimiento
col_train_labels <- col[1:277, ]$Sentiment
col_test_labels <- col[278:462, ]$Sentiment

prop.table(table(col_train_labels))
#Las proporciones entre las clases de tweets muestran una tendencia hacia la reaccion positiva (0.57), con las reacciones neutrales como
#segunda opcion (0.38) y la negatividad relegada con solo (0.046)

prop.table(table(col_test_labels))
#Para el conjunto de prueba se observa una distribucion similar ( con las reacciones positiva y neutral mas cercanas 0.49 y 0.47 respectivamente, las reacciones negativas son escasas con 0.032)

wordcloud(col_corpus_clean, min.freq=10, random.order=FALSE)
#Nube de palabras de corpus

pos <- subset(col, Sentiment=="1")
neg <- subset(col, Sentiment=="-1")
neu <- subset(col, Sentiment=="0")
#Separacion de mensajes según su sentimiento

wordcloud(pos$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets positivos

wordcloud(neg$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets negativos

wordcloud(neu$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets neutros

#Terminos frecuentes
findFreqTerms(col_dtm_train, 5)
col_freq_words <- findFreqTerms(col_dtm_train, 5)

#Eliminacion terminos irrelevantes o poco frecuentes del modelo
col_dtm_freq_train <- col_dtm_train[ , col_freq_words]
col_dtm_freq_test <- col_dtm_test[ , col_freq_words]

#Funcion para indicar si los tweets contienen o no terminos frecuentes
convert_counts <- function(x){
  x<- ifelse(x>0, "Yes", "No")
}

col_train <- apply(col_dtm_freq_train, MARGIN=2, convert_counts)
col_test <- apply(col_dtm_freq_test, MARGIN=2, convert_counts)


#Modelo de prediccion
col_classifier <- naiveBayes(col_train, col_train_labels)
col_test_pred <- predict(col_classifier, col_test)
CrossTable(col_test_pred, col_test_labels, prop.chisq=FALSE, prop.t=FALSE, dnn=c('Prediccion', 'Real'))

#El desempeño del modelo es relativamente bueno, ya que logra acercarse a la distribucion real de los tweets, de los 6 casos de tweets negativos, predice 11; 
#de los 77 tweets neutrales predice 87 y de los 92 tweets positivos predice 97.

#Modelo 2
col_classifier2 <- naiveBayes(col_train, col_train_labels, laplace=1)
col_test_pred2 <- predict(col_classifier2, col_test)
CrossTable(col_test_pred2, col_test_labels, prop.chisq=FALSE, prop.t=FALSE, dnn=c('Prediccion', 'Real'))

#Agregando un estimador de Laplace, se observa que dada la escasez de tweets negativos, el modelo directamente los elimina
#creando un sesgo hacia la neutralidad y las emociones positivas, lo que dificulta el trabajo con este metodo.

