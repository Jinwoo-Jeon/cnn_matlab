% init
opt.solver.verbose = false;
model = train(model, opt, train_img, train_label);
test(model, opt, test_img, test_label);