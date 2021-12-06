function classNames = getClassNames()
% The getClassNames function returns the names of Pandaset dataset classes.
%
% Copyright 2021 The MathWorks, Inc.

names={'Car'
       'Truck'
       'Pedestrain'};

classNames = categorical(names);
end
