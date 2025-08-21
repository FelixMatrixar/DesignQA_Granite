import argparse
import os
from metrics.metrics import eval_retrieval_qa, eval_compilation_qa, eval_definition_qa, eval_presence_qa, eval_dimensions_qa, eval_functional_performance_qa

def main():
    parser = argparse.ArgumentParser(description="Optional paths for CAD evaluation inputs")

    parser.add_argument("--path_to_retrieval", type=str, default=None,
                        help="Path to csv containing retrieval data (optional)")
    parser.add_argument("--path_to_compilation", type=str, default=None,
                        help="Path to csv containing compilation data (optional)")
    parser.add_argument("--path_to_dimension", type=str, default=None,
                        help="Path to csv containing dimension data (optional)")
    parser.add_argument("--path_to_functional_performance", type=str, default=None,
                        help="Path to csv containing functional performance data (optional)")
    parser.add_argument("--path_to_definition", type=str, default=None,
                        help="Path to csv containing definition data (optional)")
    parser.add_argument("--path_to_presence", type=str, default=None,
                        help="Path to csv containing presence data (optional)")
    parser.add_argument("--save_path", type=str, default="results.txt",
                        help="Path to .txt file to save the evaluation results (default: results.txt)")

    args = parser.parse_args()

    # Check if save path already exists and ask for user confirmation
    if os.path.exists(args.save_path):
        response = input(f"File '{args.save_path}' already exists. Do you want to overwrite it? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Operation cancelled. Exiting without overwriting existing file.")
            return

    print("Arguments received:")
    print(f"  path_to_retrieval: {args.path_to_retrieval}")
    print(f"  path_to_compilation: {args.path_to_compilation}")
    print(f"  path_to_dimension: {args.path_to_dimension}")
    print(f"  path_to_functional_performance: {args.path_to_functional_performance}")
    print(f"  path_to_definition: {args.path_to_definition}")
    print(f"  path_to_presence: {args.path_to_presence}")
    
    all_subsets = []
    if args.path_to_retrieval:
        macro_avg_retrieval, all_answers_retrieval = eval_retrieval_qa(args.path_to_retrieval)
        all_subsets.append(macro_avg_retrieval)
        
    if args.path_to_compilation:
        macro_avg_compilation, all_answers_compilation = eval_compilation_qa(args.path_to_compilation)
        all_subsets.append(macro_avg_compilation)
        
    if args.path_to_definition:
        macro_avg_definition, definitions_qs_definition_avg, multi_qs_definition_avg, single_qs_definition_avg, all_answers_definition = eval_definition_qa(args.path_to_definition)
        all_subsets.append(macro_avg_definition)
        
    if args.path_to_presence:
        macro_avg_presence, definitions_qs_presence_avg, multi_qs_presence_avg, single_qs_presence_avg, all_answers_presence = eval_presence_qa(args.path_to_presence)
        all_subsets.append(macro_avg_presence)
        
    if args.path_to_dimension:
        macro_avg_accuracy_dimension, direct_dim_avg, scale_bar_avg, all_accuracies_dimension, macro_avg_bleus_dimension, all_bleus_dimension, \
                macro_avg_rogues_dimension, all_rogues_dimension = eval_dimensions_qa(args.path_to_dimension)
        all_subsets.append(macro_avg_accuracy_dimension)

    if args.path_to_functional_performance:
        macro_avg_accuracy_functional, all_accuracies_functional, macro_avg_bleus_functional, all_bleus_functional, macro_avg_rogues_functional, all_rogues_functional = eval_functional_performance_qa(args.path_to_functional_performance)
        all_subsets.append(macro_avg_accuracy_functional)

    # Write all the results to a file
    with open(args.save_path, 'w') as text_file:
        text_file.write("DESIGNQA EVALUATION RESULTS:\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write(f"OVERALL SCORE: {sum(all_subsets) / 6}\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write(f"Retrieval Score (Avg F1 BoW): {macro_avg_retrieval if args.path_to_retrieval else 'N/A'}\n")
        text_file.write(f"Compilation Score (Avg F1 Rules): {macro_avg_compilation if args.path_to_compilation else 'N/A'}\n")
        text_file.write(f"Definition Score (Avg F1 BoC): {macro_avg_definition if args.path_to_definition else 'N/A'}\n")
        text_file.write(f"Presence Score (Avg Accuracy): {macro_avg_presence if args.path_to_presence else 'N/A'}\n")
        text_file.write(f"Dimension Score (Average Accuracy): {macro_avg_accuracy_dimension if args.path_to_dimension else 'N/A'}\n")
        text_file.write(f"Functional Performance Score (Average Accuracy): {macro_avg_accuracy_functional if args.path_to_functional_performance else 'N/A'}\n")
        text_file.write("-*-" * 20 + "\n")
        text_file.write("\n\n\n")
        text_file.write("Below scores by subset are provided for diagnostic purposes:\n")
        text_file.write("---" * 20 + "\n")
        text_file.write("RETRIEVAL\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_retrieval:
            text_file.write(f"All F1 BoWs:\n{all_answers_retrieval}\n")
        else:
            text_file.write("No retrieval data provided.\n")
        
        text_file.write("---" * 20 + "\n")
        text_file.write("COMPILATION\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_compilation:
            text_file.write(f"All F1 Rules:\n{all_answers_compilation}\n")
        else:
            text_file.write("No compilation data provided.\n")
        
        text_file.write("---" * 20 + "\n")
        text_file.write("DEFINITION\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_definition:
            text_file.write(f"Avg F1 BoC on definition-components:\n{definitions_qs_definition_avg}\n")
            text_file.write(f"Avg F1 BoC on multimention-components:\n{multi_qs_definition_avg}\n")
            text_file.write(f"Avg F1 BoC on no-mention-components:\n{single_qs_definition_avg}\n")
            text_file.write(f"All F1 BoC:\n{all_answers_definition}\n")
        else:
            text_file.write("No definition data provided.\n")
            
        text_file.write("---" * 20 + "\n")
        text_file.write("PRESENCE\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_presence:
            text_file.write(f"Avg accuracy on definition-components:\n{definitions_qs_presence_avg}\n")
            text_file.write(f"Avg accuracy on multimention-components:\n{multi_qs_presence_avg}\n")
            text_file.write(f"Avg accuracy on no-mention-components:\n{single_qs_presence_avg}\n")
            text_file.write(f"All accuracies:\n{all_answers_presence}\n")
        else:
            text_file.write("No presence data provided.\n")

        text_file.write("---" * 20 + "\n")
        text_file.write("DIMENSION\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_dimension:
            text_file.write(f"Avg accuracy directly-dimensioned:\n{direct_dim_avg}\n")
            text_file.write(f"Avg accuracy scale-bar-dimensioned:\n{scale_bar_avg}\n")
            text_file.write(f"All accuracies:\n{all_accuracies_dimension}\n")
            text_file.write(f"Avg BLEU score:\n{macro_avg_bleus_dimension}\n")
            text_file.write(f"All BLEU scores:\n{all_bleus_dimension}\n")
            text_file.write(f"Avg ROUGE score:\n{macro_avg_rogues_dimension}\n")
            text_file.write(f"All ROUGE scores:\n{all_rogues_dimension}\n")
        else:
            text_file.write("No dimension data provided.\n")
            
        text_file.write("---" * 20 + "\n")
        text_file.write("FUNCTIONAL PERFORMANCE\n")
        text_file.write("---" * 20 + "\n")
        if args.path_to_functional_performance:
            text_file.write(f"All accuraciess:\n{all_accuracies_functional}\n")
            text_file.write(f"Avg BLEU score:\n{macro_avg_bleus_functional}\n")
            text_file.write(f"All BLEU scores:\n{all_bleus_functional}\n")
            text_file.write(f"Avg ROUGE score:\n{macro_avg_rogues_functional}\n")
            text_file.write(f"All ROUGE scores:\n{all_rogues_functional}\n")
        else:
            text_file.write("No functional performance data provided.\n")

if __name__ == "__main__":
    main()
    






