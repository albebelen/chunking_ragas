from pageindex import PageIndexClient
import pageindex.utils as utils
import os

FILE_PATH = "docs/CELEX_32006L0054_IT_TXT.pdf"

def page_index_chunking():
    pi_client = PageIndexClient(api_key="5949962c3b534e9ab03ae27c555050c7")

    result = pi_client.submit_document(FILE_PATH)
    doc_id = result["doc_id"]

    is_ready = False
    while is_ready != True:
        status = pi_client.get_document(doc_id)["status"]
        if status == 'completed':
            is_ready = True

    tree = pi_client.get_tree(doc_id, node_summary=True)
    print('tree length: ' + str(len(tree)))
    print(tree)
    res_list = tree.get("result")
    print('res length: ' + str(len(res_list)))
    print(res_list)
            
    # with open('page_index_output.txt', 'w') as tree_output:
    #     nodes = res_list[0]['nodes']
    #     full_text = print_nodes(nodes)
    #     tree_output.write(full_text)
    
    return tree

def print_nodes(nodes, text_to_add = ""):
    for i in nodes:
        text_to_add.append('node_id:' + i['node_id'] + '\n')
        text_to_add.append('text:' + i['text'] + '\n')

        if i['nodes'] and len(i['nodes']) > 0:
            print_nodes(i[nodes], text_to_add)
    return text_to_add