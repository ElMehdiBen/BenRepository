# Map 1-based optional input ports to variables
SS <- maml.mapInputPort(1) # class: data.frame

W=SS[lapply(SS,function(y) length(unique(y))>2) == TRUE & (!colnames(SS) %in% c('cosmos_customerid_x','cosmos_customerid_y'))]

#Threshold_specification function return threshold for each column having reel values
threshold_specification<-function(col){         
    sorted<-sort(table(W[[col]]), decreasing = TRUE) 
    return(names(sorted)[1])
}
#Feature_binarization function transform columns to binary ones   
feature_binarisation <- function(col){  
    SS[[col]]<<-ifelse(W[[col]]>threshold_specification(noquote(col)),1,0)
}   

invisible(lapply(seq_len(ncol(W)), function(x) lapply(names(W)[x], feature_binarisation)))

maml.mapOutputPort("SS");


# Input: dataset
# Output: model

library(bnlearn)

features <- get.feature.columns(dataset)
labels   <- get.label.column(dataset)
train.data <- data.frame(features, labels)
feature.names <- get.feature.column.names(dataset)
names(train.data) <- c(feature.names, "label")

bn.mmhc <- mmhc(train.data)
plot(bn.mmhc)
model <- bn.fit(bn.mmhc, method= "bayes", train.data)


# Map 1-based optional input ports to variables
test_set <- maml.mapInputPort(1) # class: data.frame

library(caret)

precision <- posPredValue(test_set$Scored_Labels, test_set$label, positive="1")
recall    <- sensitivity(test_set$Scored_Labels, test_set$label, positive="1")
F1        <- (2 * precision * recall) / (precision + recall)


metrics <- data.frame(Metric=character(), Value=double()) 
metrics <- data.frame()

de <- list(Metric=c("Precision","Recall","F1 Score"), Value=c(precision,recall,F1))
metrics = rbind(metrics,de)

# Select data.frame to be sent to the output Dataset port
maml.mapOutputPort("metrics");