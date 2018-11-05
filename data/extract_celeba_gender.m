clear all;
low_age=10;
high_age=80;
fin = fopen('list_attr_celeba.txt','r');
fout = fopen('celeba_genderage.txt','w');
line = fgets(fin); %skip num
line = fgets(fin); %skip attr name
while 1
    line = [];
    line = fgets(fin);
    if line(1) == '\n'
        break;
    end
    if isempty(line)
        break;
    end
    [filename, line] = strtok(line,' ');
    if isempty(line)
        break;
    end
    val = sscanf(line,'%d',[1, inf]);
    if size(val(:),1) ~= 40
        break;
    end
    fullname = sprintf('celeba-112X112\\img_align_celeba\\%s',filename);
    if exist(fullname,'file')
        if val(21) == 1
            label = 1;
        else
            label = 0;
        end
        fprintf(fout,'data\\%s %d',fullname,label);
        for i=low_age:high_age
            fprintf(fout,' -1');
        end
        fprintf(fout,'\n');
    end
end
fclose(fin);
fclose(fout);