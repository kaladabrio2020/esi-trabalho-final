import os
import ast
import csv
import tokenize

# Tipos de nó que contam como decisões para complexidade ciclomática
DECISION_NODES = (
    ast.If,
    ast.For,
    ast.While,
    ast.Try,
    ast.With,
    ast.BoolOp,
    ast.IfExp,
)

def count_decision_points(tree: ast.AST) -> int:
    """
    Conta nós de decisão no AST para calcular complexidade ciclomática.
    Cada BoolOp (and/or) adiciona decisões extras por cada operador lógico.
    """
    cnt = sum(isinstance(n, DECISION_NODES) for n in ast.walk(tree))
    cnt += sum(len(n.values) - 1 for n in ast.walk(tree)
               if isinstance(n, ast.BoolOp))
    return cnt + 1  # McCabe

def max_ast_depth(tree: ast.AST) -> int:
    """
    Calcula a profundidade máxima da árvore AST.
    """

    def visit(node: ast.AST, depth: int = 0):
        depths = [depth]
        for child in ast.iter_child_nodes(node):
            depths.append(visit(child, depth + 1))
        return max(depths)

    return visit(tree)

def count_comments_and_blanks(path: str) -> tuple[int, int]:
    """
    Conta comentários e linhas em branco usando o módulo tokenize.
    Comentários são detectados por tokens COMMENT.
    """
    comments = blanks = 0

    # Contar comentários
    with open(path, 'rb') as f:
        tokens = tokenize.tokenize(f.readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                comments += 1

    # Contar linhas em branco
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip():
                blanks += 1

    return comments, blanks

def analyze_file(path: str) -> dict:
    """
    Analisa um arquivo .py, extrai diversas métricas e retorna um dicionário com os resultados.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        src = f.read()

    tree = ast.parse(src)

    # Métricas básicas
    loc = sum(1 for l in src.splitlines() if l.strip())
    comments, blanks = count_comments_and_blanks(path)

    # Coleta funções e classes
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    # Estatísticas de funções e classes
    n_funcs = len(funcs)
    n_classes = len(classes)

    avg_params = (sum(len(f.args.args) for f in funcs) / n_funcs) if n_funcs else 0.0

    methods_per_class = [
        sum(isinstance(n, ast.FunctionDef) for n in cls.body)
        for cls in classes
    ]

    avg_methods = (sum(methods_per_class) / n_classes) if n_classes else 0.0
    
    # Exceções Levantadas e Tratadas
    n_raises = sum(isinstance(n, ast.Raise) for n in ast.walk(tree))
    n_except = sum(isinstance(n, ast.ExceptHandler) for n in ast.walk(tree))

    # Numero de chamadas internas e externas
    defined_funcs = {f.name for f in funcs}
    n_internal = n_external = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            func = n.func
            if isinstance(func, ast.Name) and func.id in defined_funcs:
                n_internal += 1
            else:
                n_external += 1


    # Complexidade e profundidade AST
    cyclo = count_decision_points(tree)
    depth = max_ast_depth(tree)

    return {
        'FILE': path,
        'LOC': loc, #Lines of Code
        'COM': comments, #Lines of Comments
        'BLK': blanks, #Lines of Blank
        'NOF': n_funcs, #Number of Functions
        'NOC': n_classes,#Number of Classes
        'APF': round(avg_params, 2), #Average Parameters per Function
        'AMC': round(avg_methods, 2), #Average Methods per Class
        'NER': n_raises, #Number of Exception Raises
        'NEH': n_except, #Number of Exception Handlers 
        'CYC': cyclo, #Cyclomatic Complexity
        'MAD': depth, #Max AST Depth
        'BUG': 0 
    }

def scan_directory(root: str):
    """
    Varre recursivamente um diretório e processa todos os arquivos .py.
    """
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith('.py'):
                path = os.path.join(dirpath, name)
                yield analyze_file(path)

def save_to_csv(data: list[dict], outcsv: str) -> None:
    """
    Salva a lista de dicionários em um CSV com as chaves como cabeçalho.
    """
    if not data:
        raise ValueError("Nenhum dado para salvar")

    with open(outcsv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
