'''
Licensing Information: piech@cs.stanford.edu
'''
import collections
import math
import random
import util
from engine.const import Const
from util import Belief


class ExactInference:

    # Function: Init
    # --------------
    # Constructor that initializes an ExactInference object which has numRows x numCols number of tiles.
    def __init__(self, numRows: int, numCols: int):
        # util.Belief is a class (constructor) that represents the belief for a single
        # inference state of a single car (see util.py).
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()

    '''
    # Problem a:
    # Function: Observe (update the probabilities based on an observation)
    # Params:
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #
    # Notes:
    # - use util.rowToY() and util.colToX() to convert indices into locations, and then compute distance
    # - update probability by util.pdf and self.belief.setProb()
    # - don't forget to normalize self.belief after you update its probabilities!
    '''

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        for row in range(self.belief.numRows):
            for col in range(self.belief.numCols):
                #索引转换实际位置
                x = util.colToX(col)
                y = util.rowToY(row)
                
                trueDist = math.sqrt((agentX - x) ** 2 + (agentY - y) ** 2) #计算欧氏距离
                prob = util.pdf(trueDist, Const.SONAR_STD, observedDist)    #计算发射概率
                self.belief.setProb(row, col, self.belief.getProb(row, col) * prob)  #更新belief的概率
        
        self.belief.normalize() #归一化
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE


    '''
    # Problem b:
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # Params: ..
    #
    # Notes:
    # - take self.Belief and add probability
    # - use the transition probabilities in self.transProb, which is a dictionary
    #   containing all the ((oldTile, newTile), transProb) key-val pairs that youmust consider.
    # - use the addProb and getProb methods of the Belief class to modify
    #   and access the probabilities associated with a belief.  (See util.py.)
    # - normalize and update
    # - be careful that you are using only the CURRENT self.belief distribution to compute updated beliefs.  
    '''

    def elapseTime(self) -> None:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        #创建新的空信念分布
        newBelief=util.Belief(self.belief.getNumRows(),self.belief.getNumCols(),value=0) #创建一个新的belief  
        #遍历所有（oldTile，newTile）
        #遍历新的belief的每个格子
        for row in range(newBelief.getNumRows()):
            for col in range(newBelief.getNumCols()):
                newProb=0
                #遍历旧的belief的每个格子
                for oldRow in range(self.belief.getNumRows()):
                    for oldCol in range(self.belief.getNumCols()):
                        oldTile=(oldRow,oldCol)
                        newTile=(row,col)
                        #如果旧格子和新格子之间有转移概率
                        if (oldTile,newTile) in self.transProb:
                            transProb = self.transProb[(oldTile, newTile)]
                            oldProb = self.belief.getProb(oldRow, oldCol)
                            newProb += transProb * oldProb  #累加转移概率
                newBelief.addProb(row, col, newProb)  #更新新belief的概率

        self.belief=newBelief   #更新self.belief为新的belief
        self.belief.normalize()  #归一化
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile 
    def getBelief(self) -> Belief:
        return self.belief



# Class: Likelihood Weighting Inference
# -----------------------------------
# Uses likelihood weighting sampling to approximate the belief distribution
# over a car's position on the grid.
class LikelihoodWeighting:
    NUM_SAMPLES = 200

    def __init__(self, numRows: int, numCols: int):
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if oldTile not in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(float)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize samples uniformly
        self.samples = [(random.randint(0, numRows - 1), random.randint(0, numCols - 1)) for _ in range(self.NUM_SAMPLES)]
        self.weights = [1.0 for _ in range(self.NUM_SAMPLES)]
        self.updateBelief()

    '''
    # Problem c
    # Function: Update Belief
    # params: ..
    #
    # Notes:
    # - this function is called after observation (when weights are updated)
    #   and after time elapse (when samples are moved).
    # - a util.Belief object represents the probability of being in each grid cell
    # - For each sample, you should add its weight to the corresponding tile.
    # - also, don't forget to normalize self.belief!!   
    '''
    def updateBelief(self) -> None:
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)

        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        for i,(r,c) in enumerate(self.samples): #元组迭代，遍历每个样本
            newBelief.addProb(r,c,self.weights[i])  #将每个样本的权重添加到对应的格子   
        self.belief = newBelief  #更新self.belief为新的belief
        self.belief.normalize()  #归一化


        #raise Exception("Not implemented yet")
        # END_YOUR_CODE
        

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        for i, (r, c) in enumerate(self.samples):
            x = util.colToX(c)
            y = util.rowToY(r)
            trueDist = math.sqrt((agentX - x) ** 2 + (agentY - y) ** 2)
            self.weights[i] = util.pdf(trueDist, Const.SONAR_STD, observedDist)
        self.updateBelief()

    '''
    # Problem d
    # Function: Elapse Time
    # Params: ..
    #
    # Notes:
    # - for each current sample, sample a new location according to the transition
    #   probabilities from that position. This reflects the car moving in the world.
    # - you may find util.weightedRandomChoice() useful
    # - if a sample is in a location with no outgoing transitions defined,
    #   keep the sample in place (identity transition).
    # - after updating the sample positions, recompute the belief distribution
    #   using the new sample set and current weights by calling updateBelief()
    '''

    def elapseTime(self) -> None:
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        newSamples = []
        
        for r,c in self.samples:
            oldTile = (r, c)
            
            if oldTile in self.transProbDict and self.transProbDict[oldTile]:
                newTile=util.weightedRandomChoice(self.transProbDict[oldTile])  #根据转移概率采样新的位置
            else:
                newTile=oldTile  #如果没有转移概率，则保持不变
            
            newSamples.append(newTile)  #将新的位置添加到新样本列表
        self.samples = newSamples  #更新样本
        self.updateBelief()  #更新belief分布
        
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE


    def getBelief(self) -> Belief:
        return self.belief




class ParticleFilter:
    NUM_PARTICLES = 200

    # Function: Init
    # --------------
    # Constructor that initializes an ParticleFilter object which has (numRows x numCols) number of tiles.
    def __init__(self, numRows: int, numCols: int):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in an integer-valued defaultdict.
        # Use self.transProbDict[oldTile][newTile] to get the probability of transitioning from oldTile to newTile.
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if oldTile not in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        '''
        # Problem e: initialize the particles randomly
        # Notes:
        # - you need to initialize self.particles, which  is a defaultdict 
        #   from grid locations to number of particles at that location
        # - self.particles should contain |self.NUM_PARTICLES| particles randomly distributed across the grid.
        # - after initializing particles, you must call |self.updateBelief()| to compute the initial belief distribution.
        '''
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        self.particles = collections.defaultdict(int)  #初始化粒子 defaultdict
        validTiles = list(self.transProbDict.keys())  # 获取所有可转移的位置
        for _ in range(self.NUM_PARTICLES):
            tile = random.choice(validTiles)
            self.particles[tile] += 1  #随机选择一个位置并增加粒子计数
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE
        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles| and ensures that the probabilites sum to 1
    def updateBelief(self) -> None:
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    '''
    # Problem f
    # Function: Observe:(Takes |self.particles| and updates them based on the distance observation and your position )
    # Params:
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #             
    # Notes:
    # - your reweight operation should correspond to your initialization!!
    # - update the particle distribution with the emission probability associated with the observed distance
    # - tiles with 0 probabilities (i.e. those with no particles) do not need to be updated.
    # - this makes particle filtering runtime to be O(|particles|).
    '''

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # Reweight the particles
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        weights= collections.defaultdict(float)  #创建一个新的权重分布
        for (r,c),num in self.particles.items():   #(r,c)是行列索引，oldweights是粒子数量，self.particles[(r,c)] = num,该时刻(r,c)位置的粒子数量，也可以视为该时刻该位置的权重--粒子数越多，权重越高
            x= util.colToX(c)
            y= util.rowToY(r)
            trueDist = math.sqrt((agentX - x)**2+(agentY-y)**2)  #计算真实距离
            prob=util.pdf(trueDist, Const.SONAR_STD, observedDist)  #计算发射概率
            weights[(r,c)] = prob*num     #更新权重，
        #归一化
        totalWeight=sum(weights.values())
        if totalWeight==0:
            raise ValueError("All particle weight are zero.Check!")
        for tile in weights:
            weights[tile]/=totalWeight  #归一化，使数量变为权重
            
        neWeightedParticles = collections.defaultdict(int)  #创建一个新的粒子分布
        for _ in range(self.NUM_PARTICLES):
            #根据权重重新采样粒子
            sampledTile=util.weightedRandomChoice(weights) #根据权重随机选择一个位置
            neWeightedParticles[sampledTile] += 1
            
        self.particles=neWeightedParticles
        self.updateBelief()  #更新belief分布
        
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

        # Resample the particles
        # Now we have the reweighted (unnormalized) distribution, we can now re-sample the particles from 
        # this distribution, choosing a new grid location for each of the |self.NUM_PARTICLES| new particles.
        newParticles = collections.defaultdict(int)
        for _ in range(self.NUM_PARTICLES):
            p = util.weightedRandomChoice(self.particles)
            newParticles[p] += 1  
        self.particles = newParticles
        self.updateBelief()

  
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # We have a particle distribution at current time $t$, and we want
    # to propose the particle distribution at time $t+1$. We would like
    # to sample again to see where each particle would end up using the transition model.
    #
    # Notes:
    # - Remember that if there are multiple particles at a particular location,
    #   you will need to call util.weightedRandomChoice() once for each of them!
    # - You should NOT call self.updateBelief() at the end of this function.
    def elapseTime(self) -> None:
        newParticles = collections.defaultdict(int)
        for particle in self.particles:
            for _ in range(self.particles[particle]):
                newParticle = util.weightedRandomChoice(self.transProbDict[particle])
                newParticles[newParticle] += 1
        self.particles = newParticles

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile
    def getBelief(self) -> Belief:
        return self.belief
