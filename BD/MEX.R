.
#Tweets SelMexico
mex <- read.csv("m.csv", sep = ";", stringsAsFactors = FALSE)
mex <- mex[1:435,] #Se eliminan todas las observaciones vacias
mex$Sentiment <- factor(mex$Sentiment) #Conversion a factores de variable relevante
mex$Sentiment
str(mex)
library(tm)

#Limpieza datos
txt = mex$Tweet.Text
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


mex_corpus <- VCorpus(VectorSource(txtclean))
#Creacion corpus de texto para analisis
print(mex_corpus)
mex_corpus_clean <- tm_map(mex_corpus, content_transformer(tolower))
#Transformacion de todos los tweets a minusculas, para eliminar posibles duplicados en el analisis

mex_corpus_clean <- tm_map(mex_corpus_clean, removeWords, c(stopwords("spanish"), "que", "un", "una", "por", "la", "el"))
#Remueve stopwords y palabras irrelevantes del corpus

sw <- readLines("C:/Users/NICOLAS GARZON/Downloads/Nueva carpeta (2)/BD/stopwords.es.txt",encoding="UTF-8")
sw = iconv(sw, to="ASCII//TRANSLIT")
#Archivo con nuevas stopwords en espanol

mex_corpus_clean <- tm_map(mex_corpus_clean, removeWords, sw) #Remueve stopwords faltantes
mex_corpus_clean <- tm_map(mex_corpus_clean, stripWhitespace) #Elimina espacios vacios resultantes
as.character(mex_corpus_clean[[3]])

#Tokenizacion
mex_dtm <- DocumentTermMatrix(mex_corpus_clean)
mex_dtm
#Conjuntos de entrenamiento y prueba
mex_dtm_train <- mex_dtm[1:261, ]
mex_dtm_test <- mex_dtm[262:435, ]

#Vectores de sentimiento
mex_train_labels <- mex[1:261, ]$Sentiment
mex_test_labels <- mex[262:435, ]$Sentiment

prop.table(table(mex_train_labels))
#Las proporciones entre las clases de tweets muestran una distribucion de reacciones bastante diversa con las reacciones positivas liderando (0.53), seguidas de las neutrales (0.39);
#ademas los tweets negativos se presentan en una proporcion de 0.068

prop.table(table(mex_test_labels))
#Para el conjunto de prueba se observa una distribucion algo diferente con las reacciones positivas en una proporcion de 0.33, las neutrales con 0.42 y las negativas con 0.23

wordcloud(mex_corpus_clean, min.freq=10, random.order=FALSE)
#Nube de palabras de corpus

pos <- subset(mex, Sentiment=="1")
neg <- subset(mex, Sentiment=="-1")
neu <- subset(mex, Sentiment=="0")
#Separacion de mensajes según su sentimiento

wordcloud(pos$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets positivos

wordcloud(neg$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets negativos

wordcloud(neu$Tweet.Text, scale=c(3, 0.5))
#Nube de palabras tweets neutros

#Terminos frecuentes
findFreqTerms(mex_dtm_train, 5)
mex_freq_words <- findFreqTerms(mex_dtm_train, 5)

#Eliminacion terminos irrelevantes o poco frecuentes del modelo
mex_dtm_freq_train <- mex_dtm_train[ , mex_freq_words]
mex_dtm_freq_test <- mex_dtm_test[ , mex_freq_words]

#Funcion para indicar si los tweets contienen o no terminos frecuentes
convert_counts <- function(x){
  x<- ifelse(x>0, "Yes", "No")
}

mex_train <- apply(mex_dtm_freq_train, MARGIN=2, convert_counts)
mex_test <- apply(mex_dtm_freq_test, MARGIN=2, convert_counts)


#Modelo de prediccion
mex_classifier <- naiveBayes(mex_train, mex_train_labels)
mex_test_pred <- predict(mex_classifier, mex_test)
CrossTable(mex_test_pred, mex_test_labels, prop.chisq=FALSE, prop.t=FALSE, dnn=c('Prediccion', 'Real'))

#El desempeño del modelo es bastante regular, en primer lugar sobreestima las reacciones neutrales ya que predice 123 cuando realmente son 74;
#reduce considerablemente las reacciones negativas prediciendo solo 14 de las 41 originales, y finalmente reduce los tweets positivos al predecir solo
#37 de los 59 originales
# los 7 que realmente son encontrados.

#Modelo 2
mex_classifier2 <- naiveBayes(mex_train, mex_train_labels, laplace=1)
mex_test_pred2 <- predict(mex_classifier2, mex_test)
CrossTable(mex_test_pred2, mex_test_labels, prop.chisq=FALSE, prop.t=FALSE, dnn=c('Prediccion', 'Real'))

#Agregando un estimador de Laplace, se observa que el desempeño del modelo no mejora, pues reduce las predicciones de tweets negativos (3 de 41),
#aumenta las predicciones positivas pero aun se encuentra lejos de acertar (40 de 59), y aumenta considerablemente las reacciones neutras (131 de 74). 

