#!make
# Use environment variables from .env file if it exists
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

CC ?= gcc

CFLAGS ?= -fopenmp -lm -Wall
EXEC = sudoku

all: $(EXEC)
