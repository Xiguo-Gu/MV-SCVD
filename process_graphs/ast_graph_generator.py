
import os
from typing import Dict, List

import loguru
from slither import Slither
import networkx as nx
import re
from copy import deepcopy

loguru.logger.add('file.log')

global_id_counter = 0

class Node:
    def __init__(self, node_info: Dict):
        global global_id_counter
        self.node_type = ""
        self.node_name = ""
        # 使用全局计数器作为node_id
        self.node_id = str(global_id_counter)
        global_id_counter += 1

        self.children_nodes: List[Dict] = []
        self.node_info = node_info
        self.gen_node()
        #print(self.node_name)
        if self.node_name =='':
            self.desc = f"{self.node_id} {self.node_type}"
        else:
            self.desc = f"{self.node_id} {self.node_type} {self.node_name}"

    def gen_node(self):
        for k, v in self.node_info.items():
            if k == "nodeType":
                self.node_type = v
            elif k == "name":
                self.node_name = v
            if isinstance(v, Dict) and "nodeType" in v.keys() and "id" in v.keys():
                self.children_nodes.append(v)
            if isinstance(v, List):
                for i in v:
                    if isinstance(i, Dict) and "nodeType" in i.keys() and "id" in i.keys():
                        self.children_nodes.append(i)
    def __str__(self) -> str:
        return self.desc


def deep(node: Node, graph: nx.Graph, parent_node: Node = None, root=False):
    if(node.node_name==""):
        node_name1 = None
    else:
        node_name1 = node.node_name
    graph.add_node(node.node_id,node_type=node.node_type,node_name = node_name1,source_file=file)
    if not root:
        label = get_edge_type(parent_node.node_type, node.node_type)
        #print(parent_node.node_id, node.node_name)
        graph.add_edge(parent_node.node_id, node.node_id,edge_type=label)
        #print(parent_node.node_type, node.node_type)
    for child_node in node.children_nodes:
        child = Node(child_node)
        deep(child, graph, node, root=False)


def get_edge_type(node1, node2):
    if node1 in ['IfStatement', 'WhileStatement', 'ForStatement'] and node2 in ['Block']:
        return 'ControlFlow'
    elif node1 in ['FunctionDefinition', 'VariableDeclaration'] and node2 in ['Identifier', 'UserDefinedTypeName', 'ElementaryTypeName']:
        return 'Declaration'
    elif node1 in ['Assignment', 'BinaryOperation', 'UnaryOperation'] and node2 in ['Identifier', 'Literal', 'MemberAccess', 'IndexAccess']:
        return 'Expression'
    elif node1 == 'FunctionCall' and node2 in ['Identifier', 'Literal', 'MemberAccess', 'IndexAccess']:
        return 'FunctionCall'
    elif node1 in ['IndexAccess', 'MemberAccess'] and node2 in ['Identifier', 'Literal', 'MemberAccess', 'IndexAccess']:
        return 'Access'
    elif node1 == 'TypeConversion' and node2 in ['Identifier', 'Literal', 'MemberAccess', 'IndexAccess']:
        return 'Conversion'
    elif node1 == 'Return' and node2 in ['Identifier', 'Literal', 'MemberAccess', 'IndexAccess']:
        return 'Return'
    elif node1 in ['Throw', 'Revert', 'Assert', 'Require'] and node2 in ['Identifier', 'Literal', 'MemberAccess', 'IndexAccess']:
        return 'ErrorHandling'
    elif node1 == 'ContractDefinition' and node2 == 'InheritanceSpecifier':
        return 'Inheritance'
    elif node1 == 'FunctionDefinition' and node2 == 'ModifierInvocation':
        return 'ModifierInvocation'
    elif node1 == 'ArrayTypeName' and node2 in ['Identifier', 'Literal']:
        return 'ArrayType'
    elif node1 == 'Mapping' and node2 in ['ElementaryTypeName', 'UserDefinedTypeName']:
        return 'MappingType'
    elif node1 == 'NewExpression' and node2 in ['UserDefinedTypeName', 'ArrayTypeName']:
        return 'NewExpression'
    elif node1 == 'TupleExpression' and node2 in ['Identifier', 'Literal', 'MemberAccess', 'IndexAccess', 'BinaryOperation']:
        return 'TupleExpression'
    elif node1 == 'VariableDeclarationStatement' and node2 in ['Identifier', 'Literal', 'MemberAccess', 'IndexAccess', 'BinaryOperation']:
        return 'VariableDeclarationStatement'
    elif node1 == 'Block' and node2 in ['IfStatement', 'WhileStatement', 'ForStatement', 'Block', 'ExpressionStatement', 'VariableDeclarationStatement', 'Return', 'Break', 'Continue', 'Throw']:
        return 'BlockStatement'
    elif node1 == 'ContractDefinition' and node2 in ['FunctionDefinition', 'VariableDeclaration', 'ModifierDefinition', 'EventDefinition', 'StructDefinition', 'EnumDefinition', 'UsingForDirective']:
        return 'ContractBody'
    elif node1 == 'FunctionDefinition' and node2 == 'ParameterList':
        return 'FunctionParameters'
    elif node1 == 'ModifierDefinition' and node2 == 'ParameterList':
        return 'ModifierParameters'
    else:
        return 'Other'


pattern =  re.compile(r'\d.\d.\d+')
def get_solc_version(path):
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            if 'pragma solidity' in line:
                if len(pattern.findall(line)) > 0:
                    return pattern.findall(line)[0]
                else:
                    return '0.4.25'
            line = f.readline()
    return '0.4.25'


count = 0
ROOT = './experiments/ge-sc-data/source_code'
bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
            'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50, 
            'unchecked_low_level_calls': 95}
for bug, counter in bug_type.items():
    full_graph = None
    global_id_counter = 0
    # source = f'{ROOT}/{bug}/buggy_curated'
    # output = f'{ROOT}/{bug}/buggy_curated/cfg_compressed_graphs.gpickle'
    source = f'{ROOT}/{bug}/clean_{counter}_buggy_curated_0'
    output = f'{ROOT}/{bug}/clean_{counter}_buggy_curated_0/ast'
    #print(files,len(files))
    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith(".sol"):
                path = os.path.join(root, file)
                name1 = file[0:-4]
                #print(path)
                try:
                    sc_version = get_solc_version(path)
                    solc_compiler = f'/Users/guxiguo/.solc-select/artifacts/solc-{sc_version}'
                    if not os.path.exists(solc_compiler):
                        solc_compiler = f'/Users/guxiguo/.solc-select/artifacts/solc-0.5.0'
                    #loguru.logger.info("正在处理文件: {}", path)
                    sl = Slither(path,solc=solc_compiler)
                    whole_ast = sl.crytic_compile.compilation_units[path].ast(path)
                    assert whole_ast is not None
                    contracts_node = [node for node in whole_ast['nodes'] if node["nodeType"] == "ContractDefinition"]
                    father_node = Node({
                        "id": 0,
                        "nodeType": "Father",
                        "name": None
                    })
                    father_node.children_nodes = contracts_node
                    g = nx.MultiDiGraph()
                    deep(father_node, g, root=True)
                    if full_graph is None:
                        full_graph = deepcopy(g)
                    elif g is not None:
                        full_graph = nx.disjoint_union(full_graph, g)
                       
                    #loguru.logger.info('AST树构建成功:{}',path)
                    os.makedirs(output, exist_ok=True)
                    nx.drawing.nx_pydot.write_dot(g,output+'/'+name1+'.dot')
                    #loguru.logger.info("成功处理文件: {}", path)
                except Exception as e:
                    #loguru.logger.error(f"处理文件{file}出错：{e}")
                    try:
                        solc_compiler = f'/Users/guxiguo/.solc-select/artifacts/solc-0.4.24'
                        if not os.path.exists(solc_compiler):
                            solc_compiler = f'/Users/guxiguo/.solc-select/artifacts/solc-0.5.0'
                        #loguru.logger.info("正在处理文件: {}", path)
                        sl = Slither(path,solc=solc_compiler)
                        whole_ast = sl.crytic_compile.compilation_units[path].ast(path)
                        assert whole_ast is not None
                        contracts_node = [node for node in whole_ast['nodes'] if node["nodeType"] == "ContractDefinition"]
                        father_node = Node({
                            "id": 0,
                            "nodeType": "Father",
                            "name": None
                        })
                        father_node.children_nodes = contracts_node
                        g = nx.MultiDiGraph()
                        deep(father_node, g, root=True)
                        if full_graph is None:
                            full_graph = deepcopy(g)
                        elif g is not None:
                            full_graph = nx.disjoint_union(full_graph, g)
                        os.makedirs(output, exist_ok=True)
                        nx.drawing.nx_pydot.write_dot(g,output+'/'+name1+'.dot')  
                    except Exception as e:
                        loguru.logger.error(f"处理文件{file}出错:{e}")
                        count = count+1
        nx.write_gpickle(full_graph,output+'/ast_compressed_graphs.gpickle',protocol=3)
        nx.nx_agraph.write_dot(full_graph, output+'/e.dot')
print(count)
