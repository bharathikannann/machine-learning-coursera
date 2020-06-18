clc;
close all;
clear all;
A = double(imread('compressed.png'));
X = reshape(A, size(A)(1) * size(A)(2), 3);
X(400000:400000,1:3)


