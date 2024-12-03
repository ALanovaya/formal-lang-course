import antlr4
from project.QueryGraphLangLexer import QueryGraphLangLexer
from project.QueryGraphLangParser import QueryGraphLangParser


def program_to_tree(program: str) -> tuple[antlr4.ParserRuleContext, bool]:
    sanitized_program = program.replace("<EOF>", "EOF")
    input_stream = antlr4.InputStream(sanitized_program)
    lexer = QueryGraphLangLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = QueryGraphLangParser(token_stream)
    parser.removeParseListeners()
    parse_tree = parser.prog()

    if parser.getNumberOfSyntaxErrors() > 0:
        print("Syntax errors detected.")
        return (None, False)

    return (parse_tree, True)


def tree_to_program(tree: antlr4.ParserRuleContext) -> str:
    if not tree:
        return ""

    class DefaultListener(antlr4.ParseTreeListener):
        def __init__(self):
            self.tokens = []

        def visitTerminal(self, node):
            self.tokens.append(node.getText())

    listener = DefaultListener()
    walker = antlr4.ParseTreeWalker()
    walker.walk(listener, tree)

    return " ".join(listener.tokens)


def nodes_count(tree: antlr4.ParserRuleContext) -> int:
    if not tree:
        return 0

    class NodeCountListener(antlr4.ParseTreeListener):
        def __init__(self):
            self.count = 0

        def enterEveryRule(self, ctx):
            self.count += 1

    listener = NodeCountListener()
    walker = antlr4.ParseTreeWalker()
    walker.walk(listener, tree)

    return listener.count
