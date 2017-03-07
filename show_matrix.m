function [b1,b2] = show_matrix(rho)
%GRAPH_MATRIX Graph real and imaginary parts of a matrix
%   GRAPH_MATRIX(RHO) is a helper function that displays and graphs the 
%   real and imaginary parts of the matrix using a 3D bar graph in the 
%   current figure

min_value = min([real(rho(:));imag(rho(:))]);
max_value = max([real(rho(:));imag(rho(:))]);

if(numel(rho)<100)
    display(rho);
    subplot(121);
    b1=bar3(real(rho)); 
    for i=1:numel(b1); 
        set(b1(i),'CData',get(b1(i),'ZData')); 
        set(b1(i),'FaceColor','interp'); 
    end;
    axis equal;
    ax = axis;
    ax(5:6) = [min_value max_value];
    axis(ax);
    caxis([min_value max_value]);
    title('real part');
    subplot(122);
    b2=bar3(imag(rho));
    for i=1:numel(b2); 
        set(b2(i),'CData',get(b2(i),'ZData')); 
        set(b2(i),'FaceColor','interp'); 
    end;
    axis equal;
    ax = axis;
    ax(5:6) = [min_value max_value];
    axis(ax);
    caxis([min_value max_value]);
    title('imaginary part');
    set(gcf,'Position',[100 100 640 240]);
else
    subplot(111);
    subplot('Position',[0.05 0.1 0.4 0.8]);
    imagesc(real(rho),[min_value max_value]);
    axis image;
    axis off;
    title('real part');
    subplot('Position',[0.55 0.1 0.4 0.8]);
    imagesc(imag(rho),[min_value max_value]);
    axis image;
    axis off;
    title('imaginary part');
    subplot('Position',[0.5 0.1 0.025 0.8]);
    imagesc(0,[min_value,max_value],linspace(min_value,max_value,256)',[min_value max_value]);
    axis xy;
    set(gca,'XTick',[]);
    set(gcf,'Position',[100 100 640 320]);
end
snapnow;

end

