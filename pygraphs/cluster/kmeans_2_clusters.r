#! /usr/bin/Rscript

args = commandArgs(trailingOnly=TRUE)

library(kernlab)

d <- read.table(args[1], header=FALSE, sep=",")
d <- as.matrix(d)
K <- as.kernelMatrix(d)

Clusters <- kkmeans(K, centers=strtoi(args[2]))

write.csv(Clusters@.Data, file=paste(args[1], "_result.csv", sep=""))
