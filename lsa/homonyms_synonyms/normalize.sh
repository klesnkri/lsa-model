#!/bin/bash

touch tmp

#delete "
tr -d '"' < $1 > tmp && mv tmp $1

#delete newlines
tr -d '\n' < $1 > tmp && mv tmp $1

# replaces whitespaces with space
tr "[:space:]" " " < $1 > tmp && mv tmp $1

#truncate spaces
tr -s " " < $1 > tmp && mv tmp $1
