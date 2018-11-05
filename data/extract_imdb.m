clear all;
load imdb_crop\\imdb.mat
gender = imdb.gender;
[age,~]=datevec(datenum(imdb.photo_taken,7,1)-imdb.dob); 
full_path = imdb.full_path;

low_age = 10;
high_age = 80;
num = size(gender(:),1);
fid1 = fopen('imdb_train.txt','w');
fid2 = fopen('imdb_test.txt','w');
for i = 1:num
    if ~isnan(gender(i))
        filename = sprintf('imdb-112X112\\%s',full_path{i});
        if exist(filename,'file')
            cur_age = int32(age(i));
            fprintf(fid1,'data\\%s %d ',filename, gender(i));
            fprintf(fid2,'data\\%s %d %d\n',filename, gender(i),cur_age);
            for j = low_age:high_age
                if j > cur_age
                    label = 1;
                else
                    label = 0;
                end
                fprintf(fid1, '%d ',label);
            end
            fprintf(fid1,'\n');
        end
    end
end
fclose(fid1);
fclose(fid2);