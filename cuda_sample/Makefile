SRCS := add.cpp
NAME := intro_cpp
CXX := g++

all: $(NAME)

$(NAME): $(SRCS)
	$(CXX) $(SRCS) -o $(NAME)

clean:
	rm -rf $(NAME)

re: clean all

.PHONY: all clean re
