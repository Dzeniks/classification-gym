import torch
import transformers
import sys
import anlys
from class_gym import Classification_Gym
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"You are using {device}")
    name = "Dzeniks/alberta_fact_checking"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, longest_first=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        name, return_dict=True, num_labels=2)
    
    model.to(device)
    claim = "The Germany lost the war in 1945"
    evidence = "Following Hitler's suicide during the Battle of Berlin, Germany signed the surrender document on 8 May 1945, ending World War II in Europe and Nazi Germany\n"
    model.to(device)
    x = tokenizer.encode_plus(claim, evidence, truncation="longest_first",
                              max_length=512, padding="max_length", return_tensors="pt")
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        prediction = model(**x)
    print(
        f"ArgMax: {torch.argmax(prediction.logits)}\nSoftMax: {torch.softmax(prediction.logits, dim=1)}")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"You are using {device}")

    name = "Dzeniks/roberta-fact-check"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, longest_first=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        name, return_dict=True, num_labels=2)
    model.to(device)

    def collate_fn(data):
        claims, evidences, labels = zip(
            *[(d['claim'], d['evidence'], d['label']) for d in data])
        labels = torch.tensor(labels)
        texts = [f"{c} {tokenizer.sep_token} {e}" for c,
                 e in zip(claims, evidences)]
        toks = tokenizer.batch_encode_plus(
            texts, truncation="longest_first", max_length=512, padding="max_length", return_tensors="pt")
        return toks, labels

    test_dataset = load_dataset("Dzeniks/hover", split="test")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        sampler=SequentialSampler(test_dataset),
        collate_fn=collate_fn
    )

    gym = Classification_Gym(model, tokenizer, name)
    loss_fn = torch.nn.CrossEntropyLoss()
    gym.test_sqce([test_loader], loss_fn)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"You are using {device}")

    name = "Dzeniks/alberta_fact_checking"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, longest_first=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        name, return_dict=True, num_labels=2)
    model.to(device)

    batch_size = 4

    def collate_fn(data):
        claims, evidences, labels = zip(
            *[(d['claim'], d['evidence'], d['label']) for d in data])
        labels = torch.tensor(labels)
        texts = [f"{c} {tokenizer.sep_token} {e}" for c,
                 e in zip(claims, evidences)]
        toks = tokenizer.batch_encode_plus(
            texts, truncation="longest_first", max_length=512, padding="max_length", return_tensors="pt")
        return toks, labels

    train_dataset = load_dataset("Dzeniks/hover", split="train")

    train_dataset = anlys.divide(train_dataset, 29_775)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        collate_fn=collate_fn,
    )

    test_dataset = load_dataset("Dzeniks/hover", split="test")
    test_dataset = anlys.divide(test_dataset, 50)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        sampler=SequentialSampler(test_dataset),
        collate_fn=collate_fn
    )

    val_dataset = load_dataset("Dzeniks/hover", split="test")
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        sampler=SequentialSampler(val_dataset),
        collate_fn=collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16 ,gamma=0.1)

    anlys.analyze(train_dataset)
    anlys.analyze(test_dataset)
    anlys.analyze(val_dataset)


    gym = Classification_Gym(model, tokenizer, name)
    gym.train_sqce_with_test(1, train_loader, test_loader,
                             val_loader, loss_fn, optimizer, scheduler, 200)


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        if args[1] == "train":
            train()
        elif args[1] == "test":
            test()
        elif args[1] == "example":
            example()
    else:
        test()
