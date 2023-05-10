#!/usr/bin/env bash
while getopts a:n:d:u: flag
do
    case "${flag}" in
        a) author_name=${OPTARG};;
        n) project_name=${OPTARG};;
        d) project_description=${OPTARG};;
        u) github_username=${OPTARG};;
    esac
done

echo "Author: $author_name";
echo "Project Name: $project_name";
echo "Description: $project_description";
echo "GitHub Username: $github_username";

echo "Renaming project..."

original_author="author_name"
original_project_name="project_name"
original_project_description="project_description"
original_github_username="github_username"

# avoid renaming things in .github folder
for filename in $(git ls-files | grep -v '/\.')
do
    sed -i "s/$original_author/$author_name/g" $filename
    sed -i "s/$original_project_name/$project_name/g" $filename
    sed -i "s/$original_project_description/$project_description/g" $filename
    sed -i "s/$original_github_username/$github_username/g" $filename
    echo "Renamed $filename"
done

mv src/project_name src/$project_name
echo "# $project_name

![Tests](https://github.com/$github_username/$project_name/actions/workflows/tests.yml/badge.svg?branch=main)
![Format Check](https://github.com/$github_username/$project_name/actions/workflows/format-check.yml/badge.svg?branch=main)

$project_description" > README.md

# This command runs only once on GHA!
rm -rf .github/template_flag.yml
