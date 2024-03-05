# For using GitHub repository
Create ssh keys for using github  
$ ssh-keygen -t ed25519 -C "Eric.Ren@lloydsbanking.com"

Start the ssh agent and add the newly generated key  
$ eval "$(ssh-agent -s)"  
$ ssh-add ~/.ssh/id_ed25519  

Upload the ssh public key to github Github: Settings --> SSH and GPg keys --> New SSH key

For using git  
$ git config --global user.name "Eric Ren"  
$ git config --global user.email Eric.Ren@lloydsbanking.com  
$ git config --global core.editor vi  
$ git config --global alias.lol "log --graph --decorate --pretty=oneline --abbrev-commit --all"  


To check all setting:

$ git config --list
