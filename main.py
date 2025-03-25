import os, sys, pdb
import numpy as np
import random
from tqdm import tqdm as progress_bar
import openai
from peft import get_peft_model, LoraConfig, TaskType

from torch import nn, no_grad
from torch.cuda.amp import autocast, GradScaler
from components.logger import ExperienceLogger
from components.engineer import PromptEngineer
from components.ct_generator import SoftPromptMixer
from components.soft_embedder import CausalEmbedding, Seq2SeqEmbedding
from components.privacy_tuning import PrivacyTuner

from utils.help import *
from utils.synthesize import build_data_generator, generate_data, prepare_generator
from utils.process import process_data, get_dataloader, check_cache
from utils.arguments import solicit_params
from utils.evaluate import eval_quantify, eval_qualify, run_eval, accelerated_eval, run_openai_eval
from utils.evaluate_mauve import MAUVEEvaluator
from utils.load import *
from assets.static_vars import dtype, debug_break, accelerator, CHECKPOINTS
from utils.help import gpt_chat_response, gpt_response

def run_in_context(args, model, dataset, exp_logger, engineer, ontology):
  if args.model == "api": # openai api
    assert (args.openai_key is not None)
    if args.verbose:
      print(f'the length of the dataset is {len(dataset)}')

    all_inputs, all_outputs, all_targets = [], [], []
    engineer.attach_dataset(args.domain, dataset)
    prompt = engineer.generate_standard_exemplars(args, ontology)
    if args.verbose:
      print("\n")
      print(f"{prompt}")
    count = 0
    except_count = 0
    for example in progress_bar(dataset, total=len(dataset)):
      all_targets.append(example['target'])
      all_inputs.append(example['text'])
      query = f"Q: {example['text']}\n     A: "
      final_prompt = prompt + query
      if args.size in ["large", "giant"]:  # gpt4 / gpt3.5
        response = gpt_chat_response(args, final_prompt)
      else:  # text-curie, text-da-vinci
        response = gpt_response(args, final_prompt)

      if args.icl_type == "base":
        all_outputs.append(response)
      elif args.icl_type == "cot":
        try:
          if args.dataset == "topv2":
            if "? " not in response.split("\n")[1]:
              attribute = ""
            else:
              attribute_start = int(response.split("\n")[1].index("?")) + 2
              attribute = response.split("\n")[1][attribute_start:]
            if "Answer: " not in response:
              slots = ""
            else:
              slots_start = int(response.index("Answer: ")) + 8
              slots = response[slots_start:]
            final_response = attribute.strip() + " <sep> " + slots.strip()
          else:
            answer_start = response.index("Answer: ")
            final_response = response[answer_start + 8:]
        except Exception:
          final_response = response
          print("No answer found!")
          print(final_response)
          except_count += 1
        all_outputs.append(final_response)
      count += 1
      if args.debug and count % 20 == 0:
        run_openai_eval(args, all_inputs, all_outputs, all_targets, exp_logger)
        print(except_count)
        break
    run_openai_eval(args, all_inputs, all_outputs, all_targets, exp_logger)

  else:
    dataloader = get_dataloader(args, dataset, 'ICL')
    num_batches = debug_break if args.debug else len(dataloader)
    exp_logger.start_eval(num_batches)
    run_eval(args, model, dataset, exp_logger)

def run_local_train(args, model, datasets, exp_logger):
  dataset, dev_dataset = datasets['train'], datasets['dev']
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  optimizer, scheduler = setup_optimization(args, model, total_steps)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()

    for step, batch in enumerate(train_dataloader):
      inputs, targets = dataset.collate(args, batch)
      review_inputs(args, inputs, targets, datasets['train'].tokenizer)

      outputs = model(**inputs, labels=targets)
      exp_logger.tr_loss += outputs.loss.item()

      if args.grad_accum_steps > 1:
        outputs.loss = outputs.loss / args.grad_accum_steps

      outputs.loss.backward()
      if (step + 1) % args.grad_accum_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

      if step % args.log_interval == 0 and step > 0:
        exp_logger.log_training(scheduler)

      if args.debug and step > debug_break:
        break

    # Evaluate on dev set
    run_eval(args, model, dev_dataset, exp_logger, 'dev')
    exp_logger.end_epoch()

def run_train_loop(args, model, datasets, exp_logger, soft_embeds=None):
  dataset, dev_dataset = datasets['train'], datasets['dev']
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  if soft_embeds is not None:
    optimizer, scheduler = setup_optimization(args, soft_embeds, total_steps)
  else:
    optimizer, scheduler = setup_optimization(args, model, total_steps)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()

    for step, batch in enumerate(train_dataloader):
      inputs, targets = dataset.collate(args, batch)
      review_inputs(args, inputs, targets, datasets['train'].tokenizer)

      outputs = model(**inputs, labels=targets)
      exp_logger.tr_loss += outputs.loss.item()

      if args.grad_accum_steps > 1:
        outputs.loss = outputs.loss / args.grad_accum_steps

      outputs.loss.backward()
      if (step + 1) % args.grad_accum_steps == 0:
        if soft_embeds is not None:
          nn.utils.clip_grad_norm_(soft_embeds.parameters(), args.max_grad_norm)
        else:
          nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

      if step % args.log_interval == 0 and step > 0:
        exp_logger.log_training(scheduler)

      if args.debug and step > debug_break:
        break

    # Evaluate on dev set
    run_eval(args, model, dev_dataset, exp_logger, 'dev')
    exp_logger.end_epoch()

  return model

def accelerated_train_loop(args, model, datasets, exp_logger, soft_embeds):
  dataset, dev_dataset = datasets['train'], datasets['dev']
  train_dataloader = get_dataloader(args, dataset)
  total_steps = len(train_dataloader) // args.grad_accum_steps * args.n_epochs
  optimizer, scheduler = setup_optimization(args, soft_embeds, total_steps)

  model, optimizer, train_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, scheduler
  )

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    model.train()

    for step, batch in enumerate(train_dataloader):
      inputs, targets = dataset.collate(args, batch)
      review_inputs(args, inputs, targets, datasets['train'].tokenizer)

      outputs = model(**inputs, labels=targets)
      exp_logger.tr_loss += outputs.loss.item()

      if args.grad_accum_steps > 1:
        outputs.loss = outputs.loss / args.grad_accum_steps

      accelerator.backward(outputs.loss)
      if (step + 1) % args.grad_accum_steps == 0:
        accelerator.clip_grad_norm_(soft_embeds.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

      if step % args.log_interval == 0 and step > 0:
        exp_logger.log_training(scheduler)

      if args.debug and step > debug_break:
        break

    # Evaluate on dev set
    accelerated_eval(args, model, dev_dataset, exp_logger, 'dev')
    exp_logger.end_epoch()

  return accelerator.unwrap_model(model)

def run_prompt_train(args, model, datasets, exp_logger, ontology):
  # freeze the large LM
  parameters = list(model.parameters())
  # can also tune the vocab embeddings by freezing first params
  # for param in parameters:
  for param in parameters:
    param.requires_grad = False

  # create and then set the soft prompt embeddings
  if args.model == 'gpt':
    soft_prompt_embed = CausalEmbedding(model.get_input_embeddings(), args.n_tokens)
  else:
    soft_prompt_embed = Seq2SeqEmbedding(model.get_input_embeddings(), args.n_tokens)
  model.set_input_embeddings(soft_prompt_embed)

  if args.accelerate:
    model = accelerated_train_loop(args, model, datasets, exp_logger, soft_prompt_embed)
  else:
    model = run_train_loop(args, model, datasets, exp_logger, soft_prompt_embed)
  return model

def run_privacy_tune(args, model, datasets, exp_logger):
  """
  使用dp-transformer对模型进行隐私微调
  
  参数:
  - args: 参数
  - model: 模型
  - datasets: 数据集
  - exp_logger: 日志记录器
  
  返回:
  - model: 微调后的模型
  """
  dataset, dev_dataset = datasets['train'], datasets['dev']
  
  # 创建隐私微调器
  privacy_tuner = PrivacyTuner(args, dataset.tokenizer)
  
  # 加载预训练模型（如果指定了预训练模型）
  if args.pretrained_model_name:
    model = privacy_tuner.load_model(args.pretrained_model_name)
  else:
    # 对现有模型应用 LoRA
    lora_config = privacy_tuner.create_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
  
  # 使用隐私微调器训练模型
  model = privacy_tuner.train(
    model=model,
    train_dataset=dataset,
    eval_dataset=dev_dataset if args.do_eval else None
  )
  
  # 保存模型
  if args.do_save:
    model_save_path = os.path.join(args.privacy_model_save_dir, f"{args.dataset}_{args.domain}")
    privacy_tuner.save_model(model, model_save_path)
  
  return model

def evaluate_with_mauve(args, generated_data, seed_data, exp_logger):
  """
  使用MAUVE评分评估生成数据
  
  参数:
  - args: 参数
  - generated_data: 生成的数据
  - seed_data: 种子数据
  - exp_logger: 日志记录器
  """
  if not args.use_mauve:
    return
  
  # 创建MAUVE评估器
  mauve_evaluator = MAUVEEvaluator(args)
  
  # 展平生成数据
  flat_generated_data = []
  for group in generated_data:
    flat_generated_data.extend(group)
  
  # 按属性评估MAUVE评分
  mauve_results = mauve_evaluator.evaluate_by_attributes(seed_data, flat_generated_data)
  
  # 记录结果
  exp_logger.log_info("MAUVE评分结果:")
  for attr, score in mauve_results.items():
    exp_logger.log_info(f"  {attr}: {score:.4f}")
  
  return mauve_results

if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args, save_path = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)
  if already_exist:
    datasets = cache_results
    ont_path = os.path.join(args.input_dir, args.dataset, "ontology.json")
    ontology = json.load(open(ont_path, 'r'))
  else:
    raw_data = load_data(args)
    ontology = raw_data['ontology']
    datasets = process_data(args, cache_results, raw_data, tokenizer)

  args.ont_size = len(ontology)
  exp_logger = ExperienceLogger(args, save_path)
  engineer = PromptEngineer(args, ontology)
  if args.method != 'dexpert' and args.model != "api":
    datasets = recruit_engineer(args, datasets, engineer)
  if args.verbose: display_domains(args, datasets)

  model = load_model(args, tokenizer, save_path)
  if args.do_train:
    if args.task == 'soft_prompt':
      run_prompt_train(args, model, datasets, exp_logger, ontology)
    elif args.task == 'privacy_tune':
      run_privacy_tune(args, model, datasets, exp_logger)
    elif args.task in ['fine_tune', 'end_to_end']:
      run_train_loop(args, model, datasets, exp_logger)
    elif args.task == 'synthesize':
      build_data_generator(args, model, datasets, exp_logger, ontology)

  elif args.do_eval and args.task != 'synthesize':
    if args.task == 'soft_prompt':
      model = load_best_soft_prompt(args, model, exp_logger)
    else:
      model = load_best_model(args, exp_logger, tokenizer)
    run_eval(args, model, datasets['test'], exp_logger, 'test')

  elif args.task == 'in_context':
    engineer.embed_samples(datasets['test'])
    run_in_context(args, model, datasets['test'], exp_logger, engineer, ontology)

  elif args.task == 'synthesize':
    if args.model == 'aug':
      model = load_pretrained_model(args, args.checkpoint)
      tokenizer = load_pretrained_tokenizer(args.method)
      generator = prepare_generator(args, model, tokenizer, exp_logger, engineer, ontology)
      generated_data = generate_data(args, generator, datasets['train'], exp_logger)
    elif args.model == 'api':
      for split, dataset in datasets.items():
        engineer.attach_dataset(args.domain, dataset)
      generator = {}
      generated_data = generate_data(args, generator, datasets['train'], exp_logger, engineer, ontology)
    else:
      generator = prepare_generator(args, model, tokenizer, exp_logger, engineer, ontology)
      generated_data = generate_data(args, generator, datasets['train'], exp_logger)
      
      # 使用MAUVE评分评估生成数据
      if args.use_mauve:
        evaluate_with_mauve(args, generated_data, datasets['train'], exp_logger)
