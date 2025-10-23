pwd # get current directory
ls # list content of directory
ls -l # more options
cd # go to specified location
clear # clear terminal screen display
touch # create a file
rm # delete a file
mkdir # create a directory
rmdir # delete a empty directory
rm -r # delete a non-empty directory, -r is short form, --recursive is long version

echo 'hello' # print msg in the command line
echo 'hello' > hello.txt # write string into file # echo will create the new file if not exist. '>' will replace all existing content
echo 'hoho' >> hello.txt # append new text in a new line

cat hello.txt # read the file to the terminal
cat hello.txt > hello_copy.txt # '>' overwrite
cat hello.txt >> hello_copy.txt # '>>' append

## find basics
find [path] [option] [expression]
find -name 'forest*' # find based on name
find -iname 'forest*' # case insensitive
find -type f / -type d # file or directory
find -type f -name 'forest*'

## move and rename
mv 'old_name' 'new_name'
mv [name] [new location] # WARNING: if new location does not exist, it will be renamed as 'new_location'

## copy
cp [original_name] [copy_name]
cp -r [original_dir] [copy_dir]

## search files
grep [option] [pattern] [file(s)] # space in between multi-files
options: 
-n # show line number
-i # case insensitive
-r # recursive in searching directory

## replace content
# default behavior: only first instance is replaced. The file itself is not modified. Only printed output contains the replacement
sed 's/pattern/replacement/' [filename]
sed 's/,/:/' team*

sed 's/pattern/replacement/[option (s)]' [filename]
sed 's/a/z/gI' team* # 'g' option will replace all instances. 'I' will make it case insensitive

# To replace original file, use '>' or '-i'. '>' is safer
sed 's/a/z/' team* > new_team.txt
sed -i 's/a/z/' team* # Bash
sed -i '' 's/a/z/' team* # zsh (mac)





