def evaluate_class_error(testloader, classes_def, nClasses)
    class_correct = list(0. for i in range(nClasses))
    class_total = list(0. for i in range(nClasses))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


for i in range(nClasses):
    print('Accuracy of %5s : %2d %%' % (
        classes_def[i], 100 * class_correct[i] / class_total[i]))
