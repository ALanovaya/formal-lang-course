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

    output = []
    for child in tree.children:
        if isinstance(child, antlr4.tree.Tree.TerminalNodeImpl):
            output.append(child.getText())
        elif isinstance(child, antlr4.ParserRuleContext):
            output.append(tree_to_program(child))

    return " ".join(output)


def nodes_count(tree: antlr4.ParserRuleContext) -> int:
    if not tree or not hasattr(tree, "children"):
        return 1

    total_nodes = 1
    for child in tree.children:
        if isinstance(child, antlr4.ParserRuleContext):
            total_nodes += nodes_count(child)
    return total_nodes
