
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            