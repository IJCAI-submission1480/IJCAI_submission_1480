trimap_path = '/path/to/trimap';
pred_path = '/path/to/predictt/result';
alpha_path = '/path/to/gt/alpha';

alpha_list = dir(alpha_path);
alpha_list = alpha_list(3:end);

total_sad_loss = 0;
total_mse_loss = 0;
total_grad_loss = 0;
total_conn_loss = 0;

img_size = [800, 800];

for i = 1:length(alpha_list)
    target = imread([alpha_path, '/', alpha_list(i).name]);
    target = imresize(target, img_size);
    target = target(:, :, 1);
    
    for j = 1:20
        % trimap = imread([trimap_path, '/', alpha_list(i).name(1:end-4), '_', num2str(j-1), '.png']);
        trimap = ones(img_size) * 128;
        pred = imread([pred_path, '/', alpha_list(i).name(1:end-4), '_', num2str(j-1), '.png']);
        pred = imresize(pred, img_size);
        pred = pred(:,:,1);
        
        sad = compute_sad_loss(pred,target,trimap);
        mse = compute_mse_loss(pred,target,trimap);
        grad = compute_gradient_loss(pred,target,trimap)/1000;
        conn = compute_connectivity_error(pred,target,trimap,0.1)/1000;
        
        total_sad_loss = total_sad_loss + sad;
        total_mse_loss = total_mse_loss + mse;
        total_grad_loss = total_grad_loss + grad;
        total_conn_loss = total_conn_loss + conn;
        
        disp([alpha_list(i).name, num2str(j-1), ' SAD: ', num2str(sad), ' MSE: ', num2str(mse), ' GRAD: ', num2str(grad), ' CONN: ', num2str(conn)]);
    end
end

image_num = length(alpha_list) * 20;
disp(['MEAN: ', ' SAD: ', num2str(total_sad_loss/image_num), ' MSE: ', num2str(total_mse_loss/image_num), ' GRAD: ', num2str(total_grad_loss/image_num), ' CONN: ', num2str(total_conn_loss/image_num)]);

