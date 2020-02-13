
# coding: utf-8

# In[71]:


class DSU():
    
    def __init__(self, lst):
        '''
        
        
        '''
        self.d = {elem:idx for idx, elem in enumerate(lst)}
        
    def is_union(self,a,b):
        if self.d[a] == self.d[b]:
            return True
        return False
  
        
    def join(self,a,b):
        self.d[b] = self.d[a]
        
    def add(self,elem):
        k = len(self.d)
        self.d[elem] = k
        
    def __len__(self):
        return len(self.d)
        
    def __getitem__(self, elem):
        return self.d[elem]
            
    def __str__(self):
        return str(self.d)
    
    def __repr__(self):
        return str(self.d.keys())


# In[72]:


setion = DSU([1,2,3])
setion.add(8)
print(setion.d)


# In[73]:


print(setion.is_union(2,3))
setion.join(2,3)

print(setion.is_union(2,3))
print(setion.d)


# In[74]:


print(len(setion))
print(setion)
print(setion[1])

