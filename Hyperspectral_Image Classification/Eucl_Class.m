classdef Eucl_Class
    properties (SetAccess=private)
        input
        target
    end
    methods
        %In this constructor we are basically fitting the classifier
        function obj=Eucl_Class(x,y)
            obj.input=x(:,:);
            obj.target=y(:,:);
        end  
        
        function r= predict(obj,test)
            r=[];
            mn_values=cell(1,5);
            Part=cell(1,5);
            for i=1:5
                Part{1,i}={};
            end    
            for i=1:size(obj.input,1)
                Part{1,obj.target(1,i)}=[Part{1,obj.target(1,i)}; obj.input(i,:)];
            end
            %Here we are compute the mean vectors of each class
            for i=1:5
                mn_values{1,i}=mean(cell2mat(Part{1,i}));
            end
            %And here we are making the predictions for every object of the
            %test
            for i=1:size(test,1)
                dist=[];
                %We compute the euclidean distance of the object with the
                %mean vectors
                for j=1:size(mn_values,2)
                    dist(j)=norm(mn_values{1,j}-test(i,:));
                end
                %Finding the index of the minimum distance and passing it to predictions array 
                [~,I]=min(dist);
                r(i)=I;
            end
            r=r';
        end 
    end   
end